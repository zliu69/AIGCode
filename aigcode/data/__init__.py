from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from torch.utils.data import DataLoader, DistributedSampler
import os
import random

from ..aliases import PathOrStr
from ..config import DataConfig, TrainConfig
from ..exceptions import AIGCodeConfigurationError
from ..torch_util import barrier, get_global_rank, get_world_size, is_distributed
from .collator import DataCollator
from .iterable_dataset import IterableDataset
from .memmap_dataset import MemMapDataset

__all__ = ["MemMapDataset", "DataCollator", "IterableDataset", "build_eval_dataloader", "build_train_dataloader"]


def build_memmap_dataset(
    train_config: TrainConfig, data_config: DataConfig, include_instance_metadata: bool = True
) -> MemMapDataset:
    paths: List[str] =[]
    metadata: List[Dict[str, Any]] = []
    metadata_paths_stats = {}
    paths = []
    label_mask_paths = []
    if data_config.paths:
        if data_config.datasets:
            raise AIGCodeConfigurationError("DataConfig.paths is mutually exclusive with DataConfig.datasets")
        paths = data_config.paths
        for path in paths:
            metadata.append({"path": str(path)})
            dataset_name = "_".join(path.split("/")[-4:-1])
            if dataset_name not in metadata_paths_stats:
                metadata_paths_stats[dataset_name] = 0.0

    elif data_config.data_dir != None and len(data_config.data_dir) > 0:
        print("tokens or max_duration(steps):", train_config.max_duration)
        
        for di in data_config.data_dir:
            for dir, data_ratio in di.items():

                print("dir: {} \n ratio: {} \n".format(dir, data_ratio))
                pdir = data_config.data_dir_prefix + "/" + dir 
                if data_ratio > 1:
                    data_ratio = int(data_ratio)
                    for _ in range(1, data_ratio):
                        for root, dirs, files in os.walk(pdir):
                            for f in files:
                                if not f.endswith(".npy") or f.endswith("_label_mask.npy"):
                                    continue
                                fpath = str(os.path.join(root, f)) 
                                metadata.append({"path": fpath})
                                dataset_name = "_".join(fpath.split("/")[-4:-1])
                                if dataset_name not in metadata_paths_stats:
                                    metadata_paths_stats[dataset_name] = 0.0
                                paths.append(fpath)
                                f_lm = f.strip(".npy") + "_label_mask.npy"
                                if f_lm in files:
                                    label_mask_paths.append(str(os.path.join(root, f_lm)))
                              
                else:
                    for root, dirs, files in os.walk(pdir):
                        fi = 0 
                        for f in files:
                            fi += 1
                            
                            if (f.endswith(".npy") and not f.endswith("_label_mask.npy")) and random.random() < data_ratio:
                                fpath = str(os.path.join(root, f)) 
                                metadata.append({"path": fpath})
                                dataset_name = "_".join(fpath.split("/")[-4:-1])
                                if dataset_name not in metadata_paths_stats:
                                    metadata_paths_stats[dataset_name] = 1.0
                                else:
                                    metadata_paths_stats[dataset_name] = metadata_paths_stats[dataset_name] + 1
                                paths.append(fpath)
                                f_lm = f.strip(".npy") + "_label_mask.npy"
                                if f_lm in files:
                                    label_mask_paths.append(str(os.path.join(root, f_lm)))
                                # data_config.file_count += 1

    elif data_config.datasets:
        paths = []
        for label in sorted(data_config.datasets.keys()):
            label_paths = data_config.datasets[label]
            paths.extend(label_paths)
            metadata.extend([{"label": label}] * len(label_paths))
    else:
        raise AIGCodeConfigurationError("One of DataConfig.paths/DataConfig.data_dir or DataConfig.datasets is required")

    print("metadata len:{}".format(len(metadata)))
    print("metadata_paths_stats: {}".format(metadata_paths_stats))
    print("label_mask_paths: {}".format(label_mask_paths))
    
    return MemMapDataset(
        *paths,
        chunk_size=train_config.model.max_sequence_length,
        memmap_dtype=data_config.effective_memmap_dtype,
        metadata=metadata,
        include_instance_metadata=include_instance_metadata,
        pad_token_id=train_config.model.pad_token_id,
        generate_attention_mask=data_config.generate_attention_mask,
        label_mask_paths=label_mask_paths,
        instance_filter_config=data_config.instance_filter,
    )


def build_eval_dataloader(
    train_config: TrainConfig,
    data_config: DataConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = build_memmap_dataset(train_config, data_config, include_instance_metadata=True)
    collator = DataCollator(pad_direction=data_config.pad_direction, pad_token_id=train_config.model.pad_token_id)
    if data_config.drop_last:
        # Make sure batch size is small enough.
        samples_per_device = len(dataset) // get_world_size()
        batch_size = min(batch_size, samples_per_device)
        assert batch_size > 0, f"dataset for {data_config.paths} is too small"
    seed = data_config.seed if data_config.seed is not None else train_config.seed
    sampler = DistributedSampler(
        dataset,
        drop_last=data_config.drop_last,
        shuffle=shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=data_config.num_workers,
        sampler=sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=None if data_config.num_workers == 0 else data_config.prefetch_factor,
        persistent_workers=False if data_config.num_workers == 0 else data_config.persistent_workers,
        timeout=data_config.timeout,
    )


def build_train_dataloader(
    train_config: TrainConfig,
    *,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    fs_local_rank: Optional[int] = None,
    include_instance_metadata: bool = False,
) -> DataLoader:
    assert train_config.device_train_batch_size is not None
    collator = DataCollator(
        pad_direction=train_config.data.pad_direction, pad_token_id=train_config.model.pad_token_id
    )
    dataset = build_memmap_dataset(
        train_config, train_config.data, include_instance_metadata=include_instance_metadata
    )
    print("dataset len:{}".format(len(dataset)))
    work_dir = Path(train_config.save_folder) / "train_data"
    if get_global_rank() == 0:
        if work_dir.is_dir() and not train_config.save_overwrite:
            raise AIGCodeConfigurationError(
                "train data working directory already exists, use --save_overwrite to overwrite"
            )
        else:
            work_dir.mkdir(exist_ok=True, parents=True)
    barrier()
    seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
    return DataLoader(
        IterableDataset(
            dataset,  # type: ignore
            train_config.global_train_batch_size,
            seed=seed + (train_config.epoch or 0),
            shuffle=True,
            drop_last=train_config.data.drop_last,
            world_size=world_size,
            rank=rank,
            fs_local_rank=fs_local_rank,
            work_dir=work_dir,
        ),
        batch_size=train_config.device_train_batch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=collator,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )
