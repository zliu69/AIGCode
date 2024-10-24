from __future__ import annotations

import cProfile
import gc
import logging
import math
import os
import random
import shutil
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from pstats import SortKey
from typing import Any, Callable, Deque, Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .aliases import PathOrStr
from .checkpoint import Checkpointer, FullCheckpointer, build_sharded_checkpointer
from .config import (
    CheckpointType,
    DDPGradSyncMode,
    DistributedStrategy,
    SchedulerUnits,
    ShardedCheckpointerType,
    SpeedMonitorConfig,
    TrainConfig,
)
from .data import IterableDataset
from .eval import Evaluator
from .exceptions import AIGCcodeConfigurationError
from .model import AIGCcode
from .optim import Optimizer, Scheduler
from .torch_util import (
    barrier,
    gc_cuda,
    get_fs_local_rank,
    get_global_rank,
    get_world_size,
    move_to_device,
    peak_gpu_memory,
    synchronize_flag,
    synchronize_value,
)
from .util import upload

__all__ = ["TrainEvaluator"]

log = logging.getLogger(__name__)

def cross_entropy_loss(
    logits, labels, ignore_index: int = -100, reduction: str = "mean", compute_z_loss: bool = False
):
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2)
    if reduction == "mean":
        z_squared = (z_squared * (labels != ignore_index)).mean()
    elif reduction == "sum":
        z_squared = (z_squared * (labels != ignore_index)).sum()

    z_loss = 1e-4 * z_squared

    return loss, z_loss

@dataclass
class TrainEvaluator:
    cfg: TrainConfig
    model: AIGCcode
    dist_model: Union[DDP, FSDP]
    evaluators: List[Evaluator]
    device: torch.device
    indices_file: Optional[TextIO] = None
    _gc_init_state: Optional[bool] = True
    loss_fn: Callable[..., torch.Tensor] = field(default_factory=lambda: cross_entropy_loss)

    def restore_unsharded_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
    ):
        # Zero-gradients to avoid gathering them.
        # self.optim.zero_grad(set_to_none=True)
        checkpointer = FullCheckpointer(self.cfg)
        trainer_state = checkpointer.restore_checkpoint(
            load_path,
            self.dist_model,
            local_cache=local_cache,
            load_optimizer_state=False,
        )
        barrier()

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        *,
        checkpoint_type: Optional[CheckpointType] = None,
        local_cache: Optional[PathOrStr] = None,
    ):
        if checkpoint_type == CheckpointType.unsharded or (
            checkpoint_type is None and str(load_path).rstrip("/").endswith("-unsharded")
        ):
            self.restore_unsharded_checkpoint(
                load_path,
                local_cache=local_cache,
            )
        elif checkpoint_type is not None:
            raise NotImplementedError(checkpoint_type)

        gc_cuda()

    def remove_checkpoint(self, idx: int = 0, checkpoint_type: CheckpointType = CheckpointType.sharded):
        if checkpoint_type == CheckpointType.sharded:
            self.remove_sharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.unsharded:
            self.remove_unsharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            self.remove_ephemeral_checkpoint(idx=idx)
        else:
            raise NotImplementedError(checkpoint_type)

    def get_labels(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, label_mask, attention_mask, instance_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
            batch.get("instance_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, -100)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, -100)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
        return labels[..., 1:].contiguous()

    def model_forward(
        self, batch: Dict[str, Any], loss_reduction: str = "mean", compute_z_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        # shape: (batch_size, seq_len, vocab_size)
        logits, _, _, l_aux, meta_data = self.dist_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            attention_bias=batch.get("attention_bias"),
        )
        # ).logits
        logits_for_loss = logits[..., :-1, :].contiguous()
        # shape: (batch_size * seq_len, vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len)
        labels = self.get_labels(batch)
        # shape: (batch_size * seq_len,)
        labels = labels.view(-1)
        ce_loss, z_loss = self.loss_fn(
            logits_for_loss, labels, ignore_index=-100, reduction=loss_reduction, compute_z_loss=compute_z_loss
        )
        if loss_reduction == "none":
            # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
            ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
            if z_loss is not None:
                z_loss = z_loss.view(batch["input_ids"].shape[0], -1)
        return ce_loss, z_loss, logits, l_aux, meta_data

    def eval_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
            ce_loss, _, logits, _, meta_data = self.model_forward(batch, loss_reduction="none")
        return ce_loss.mean(dim=-1), logits, meta_data

    def eval_step(self, batch: Dict[str, Any], evaluator: Evaluator):
        metrics: Dict[str, float] = {}
        
        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward pass.
        with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
            ce_loss, logits, meta_data = self.eval_batch(batch)

        # Update metrics.
        evaluator.update_metrics(
            batch, ce_loss, logits
        )  # batch includes all keys that the downstream evaluation needs

        barrier()

        if meta_data is not None and len(meta_data.keys())>0:
            
            for k,v in meta_data.items():
                if type(v) == list:
                    dist.reduce(meta_data[k], 0)
                    meta_data[k].div_(get_world_size())
                else:
                    continue
        # if meta_data is not None and len(meta_data.keys())>0:
            for k in meta_data.keys():
                if meta_data[k] is not None:
                    metrics["train/{}".format(k)] = meta_data[k].tolist()
        # print("metrics: {}\n".format(metrics))
        return metrics

    def log_metrics_to_console(self, prefix: str, metrics: Dict[str, float]):
        def format_float(value) -> str:
            if type(value) == float:
                if value < 0.0001:
                    return str(value)  # scientific notation
                elif value > 1000:
                    return f"{int(value):,d}"
                elif value > 100:
                    return f"{value:.1f}"
                elif value > 10:
                    return f"{value:.2f}"
                elif value > 1:
                    return f"{value:.3f}"
                else:
                    return f"{value:.4f}"
            else:
                return "{}".format(value)

        log.info(
            f"{prefix}\n"
            + "\n".join(
                [
                    f"    {name}={format_float(value)}"
                    for name, value in metrics.items()
                    if name == "optim/total_grad_norm"
                    or not name.startswith("optim/")  # there's too many optimizer metrics
                ]
            )
        )

    def eval(self) -> Dict[str, Any]:
        # Zero gradients and set model to 'eval' mode.
        # self.optim.zero_grad(set_to_none=True)
        self.dist_model.eval()

        eval_metrics = {}
        metrics_acc_all = {}
        for evaluator in self.evaluators:
            log.info(f"Running evaluation for '{evaluator.label}'...")
            metrics_acc = {}
            # Reset metrics.
            evaluator.reset_metrics()

            # Initialize data loader iterator.
            eval_batches = iter(evaluator.eval_loader)
            print("len eval_batches : {}\n".format(len(eval_batches)))
            # Adjust how many batches to evaluate on.
            num_eval_batches = (
                evaluator.subset_num_batches
                if evaluator.subset_num_batches is not None
                else self.cfg.eval_subset_num_batches
            )
            if num_eval_batches > 0:
                num_eval_batches = min(num_eval_batches, len(evaluator.eval_loader))
                eval_batches = islice(eval_batches, num_eval_batches)

            # Run model over batches.
            for eval_step, eval_batch in enumerate(eval_batches):
                # # self.eval_step(eval_batch, evaluator)
                # print("eval_step: {}".format(eval_step))
                # print("eval_batch: {}".format(eval_batch))
                # print("eval_batch shape: {}".format(eval_batch["input_ids"].shape))
                # Run train step on batch.
                metrics = self.eval_step(eval_batch, evaluator)

                for k, v in metrics.items():
                    if type(v) == list:
                        k_new = "{}_avg".format(k)
                        k_new_std = "{}_std".format(k)
                        

                        if k_new_std not in metrics_acc:
                            metrics_acc[k_new_std] = [0.0 for _ in v]
                        else:
                            metrics_acc[k_new_std] = [eval_step * metrics_acc[k_new_std][i] / (eval_step + 1) + eval_step * ((v[i] - metrics_acc[k_new][i]) ** 2) / ((eval_step + 1) ** 2) for i in range(len(v))]


                        if k_new not in metrics_acc:
                            metrics_acc[k_new] = v
                        else:
                            metrics_acc[k_new] = [(metrics_acc[k_new][i] * (eval_step) + v_acc)/(eval_step + 1) for i, v_acc in enumerate(v)]


                    else:
                        # continue
                        k_new = "{}_avg".format(k)
                        if k_new not in metrics_acc:
                            metrics_acc[k_new] = v
                        else:
                            metrics_acc[k_new] = (metrics_acc[k_new] * (eval_step) + v)/(eval_step + 1)
                # Log to console.
                if eval_step + 1 == num_eval_batches or (eval_step + 1) % self.cfg.console_log_interval == 0:
                    log.info(f"[eval_step={eval_step + 1}/{num_eval_batches}]")
                    # log.info()
                    # if len(metrics_acc.keys()) > 0:
                    #     print("metrics_acc: {} \n".format(metrics_acc))
            # Get final metrics.
            metrics = evaluator.compute_metrics()
            eval_metrics.update(metrics)
            # if metrics_acc is not None and len(metrics_acc.keys())>0:
            #     for k in metrics_acc.keys():
            #         # if type(meta_data[k]) == dict:
            #         #     for d_k in meta_data[k].keys():
            #         #         dist.reduce(meta_data[k][d_k], 0)
            #         #         meta_data[k][d_k].div_(get_world_size())
            #         # else:
            #         dist.reduce(metrics_acc[k], 0)
            #         metrics_acc[k].div_(get_world_size())
            if get_global_rank() == 0:
                print("label: {}, eval_metrics_acc: {}".format(evaluator.label, metrics_acc))

            metrics_acc_all[evaluator.label] = metrics_acc
            self.log_metrics_to_console(f"{evaluator.label}", metrics)

            del eval_batches

        return eval_metrics, metrics_acc_all

    def fit(self):
        self._gc_init_state = gc.isenabled()  # cache if garbage collection is enabled, reset on close.

        # Disable automatic garbage collection, FSDP doesn't work well with it.
        if self.cfg.gen1_gc_interval is not None:
            gc.disable()

        eval_metrics, metrics_acc_all = self.eval()
        if get_global_rank() == 0:
            print("eval_metrics: {}\n\n".format(eval_metrics))
            print("metrics_acc_all: {}\n\n".format(metrics_acc_all))

    def close(self, exit_code: int = 0) -> None:
        gc_cuda()

        if self.indices_file is not None:
            self.indices_file.flush()
            self.indices_file.close()
        if self._gc_init_state:
            gc.enable()
        else:
            gc.disable()
        if wandb.run is not None:
            wandb.finish(exit_code=exit_code, quiet=True)

    def __enter__(self) -> TrainEvaluator:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_val, exc_tb
        self.close(0 if exc_type is None else 1)
