import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
import math
from torch import Tensor
from torch.cuda import Event as CudaEvent
from torch.nn import Module, ModuleList
from .config import (
    ModelConfig,
)
from .util import (
    set_torch_seed,
)
from .torch_util import(
    get_moe_group,
    get_all2all_group,
    get_group_world_size,
    all_reduce,
)
import torch.distributed as dist
from .initialization import init_normal
from .top2gate import Top2Gate
from .top1gate import Top1Gate

class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5

log = logging.getLogger(__name__)
has_tutel = False
fused_cumsum_sub_one = lambda mask: torch.cumsum(mask, dim=0) - 1

class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output, input, group=group)
        else:
            assert group is None
            output = input
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.
def get_global_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

class FeedForwardNetwork(nn.Module):
    """
        Feed Forward Network layer in the Transformer model
    """
    def __init__(self, config, embed_dim, ffn_dim, dropout_module=None):
        super().__init__()
        self.hidden_size = config.d_model
        self.intermediate_size = (
            int(config.intermediate_size / config.exp_dim_ratio) if config.intermediate_size > 0 else int( config.mlp_ratio * config.d_model / config.exp_dim_ratio)
        ) 
        # Share MLP layer
        self.act = Activation.build(config)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(int(self.act.output_multiplier * self.intermediate_size), self.hidden_size, bias=False)
        if config.layer_share_mlp_version == 1:
            self.up_proj = nn.Linear(self.hidden_size, int(self.act.output_multiplier * self.intermediate_size), bias=False)
        self.config = config
        self.reset_params = False
        if self.config.gshard:
            print("reset params already at gpu: {}".format(get_global_rank()))
            self.reset_parameters()

    def forward(self, x):
        if self.config.layer_share_mlp_version == 1:
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        else:
            return self.down_proj(self.act(self.gate_proj(x)))

    def reset_parameters(self):
        if not self.reset_params:
            init_weights(
                self.config, self.gate_proj, d=self.hidden_size, layer_id=None, type_of_module=ModuleType.in_module
            )
            init_weights(
                self.config, self.down_proj, d=int(self.act.output_multiplier * self.intermediate_size), layer_id=None, type_of_module=ModuleType.in_module
            )
            if self.config.layer_share_mlp_version == 1:
                init_weights(
                    self.config, self.up_proj, d=self.hidden_size, layer_id=None, type_of_module=ModuleType.in_module
                )
            self.reset_params = True

def make_experts(config, embed_dim, expert_ffn_dim, dropout_module=None) -> nn.ModuleList:
    expert_list = []
    for i in range(config.moe_expert_count + config.moe_share_expert_count):
        expert_list.append(FeedForwardNetwork(config, embed_dim, expert_ffn_dim, dropout_module))
    experts = nn.ModuleList(expert_list)
    return experts


def make_experts_dist(args, embed_dim, expert_ffn_dim, dropout_module) -> nn.ModuleList:
    world_size = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
    expert_list = []
    ddp_rank = get_global_rank()
    start_seed = torch.randint(1000000, (1,)).item()
    # at least as many experts than gpus
    if args.moe_expert_count >= world_size:
        assert args.moe_expert_count % world_size == 0, f'{args.moe_expert_count}, {world_size}'
        local_moe_expert_count = args.moe_expert_count // world_size
        for i in range(local_moe_expert_count):
            with set_torch_seed(start_seed + ddp_rank * local_moe_expert_count + i):
                expert_list.append(FeedForwardNetwork(args, embed_dim, expert_ffn_dim, dropout_module))
    # less experts than gpus
    else:
        assert world_size % args.moe_expert_count == 0, f'{world_size}, {args.moe_expert_count}'
        # initialize each FFN with the same seed on different GPUs
        with set_torch_seed(start_seed + ddp_rank % args.moe_expert_count):
            print("gpu:{}, random_test with sum seed: {}".format(ddp_rank, torch.randint(0,10000, size=(1,)).item()))
            print("gpu:{}, seed: {}, new_seed:{},".format(ddp_rank, start_seed, start_seed + ddp_rank % args.moe_expert_count))
            expert_list.append(FeedForwardNetwork(args, embed_dim, expert_ffn_dim, dropout_module))
    experts = nn.ModuleList(expert_list)
    return experts


class MOELayer(Module):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self, args, gate: Optional[Module] = None, experts: Optional[Union[Module, ModuleList]] = None) -> None:
        super().__init__()
        self.args = args
        self.hidden_size = self.args.d_model
        self.intermediate_size = (
            self.args.intermediate_size if self.args.intermediate_size > 0 else self.args.mlp_ratio * self.args.d_model
        )
        if self.args.gate_level == "sample":
            if gate is not None and experts is not None:
                self.gate = gate
                if type(experts) == ModuleList:
                    self.experts = cast(ModuleList, experts)
                else:
                    self.experts = ModuleList([experts])
            else:
                if self.args.moe_top1_expert:
                    self.gate = Top1Gate(
                        self.args,
                        self.args.d_model,
                        self.args.moe_expert_count,
                        use_fp32=self.args.moe_gating_use_fp32,
                        moe_eval_capacity_token_fraction=getattr(self.args, "moe_eval_capacity_token_fraction", 0.25),
                    )
                else:
                    self.gate = Top2Gate(
                        self.args,
                        self.args.d_model,
                        self.args.moe_expert_count,
                        self.args.moe_gating_use_fp32,
                        self.args.moe_second_expert_policy,
                        self.args.moe_normalize_gate_prob_before_dropping,
                        getattr(self.args, "moe_eval_capacity_token_fraction", 0.25),
                        getattr(self.args, "moe_batch_prioritized_routing", False),
                    )
                self.experts = make_experts(self.args, self.hidden_size, self.intermediate_size, None)
                for p in self.experts.parameters():
                    p.expert = True  # type: ignore

        elif self.args.gshard:
            if self.args.moe_top1_expert:
                    self.gate = Top1Gate(
                        self.args,
                        self.args.d_model,
                        self.args.moe_expert_count,
                        use_fp32=self.args.moe_gating_use_fp32,
                        moe_eval_capacity_token_fraction=getattr(self.args, "moe_eval_capacity_token_fraction", 0.25),
                    )
            else:
                self.gate = Top2Gate(
                    self.args,
                    self.args.d_model,
                    self.args.moe_expert_count,
                    self.args.moe_gating_use_fp32,
                    self.args.moe_second_expert_policy,
                    self.args.moe_normalize_gate_prob_before_dropping,
                    getattr(self.args, "moe_eval_capacity_token_fraction", 0.25),
                    getattr(self.args, "moe_batch_prioritized_routing", False),
                )
            self.experts = make_experts_dist(self.args, self.hidden_size, self.intermediate_size, None)

            self.expert_group = get_moe_group(args.moe_expert_count)
            self.all2all_group = get_all2all_group(args.moe_expert_count)
            for p in self.experts.parameters():
                p.expert = True  # type: ignore
            self.world_size = get_group_world_size(self.expert_group)
            self.all2all_size = get_group_world_size(self.all2all_group)


        else:
            if gate is not None and experts is not None:
                self.gate = gate
                if type(experts) == ModuleList:
                    self.experts = cast(ModuleList, experts)
                else:
                    self.experts = ModuleList([experts])
            else:
                if self.args.moe_top1_expert:
                    self.gate = Top1Gate(
                        self.args,
                        self.args.d_model,
                        self.args.moe_expert_count,
                        use_fp32=self.args.moe_gating_use_fp32,
                        moe_eval_capacity_token_fraction=getattr(self.args, "moe_eval_capacity_token_fraction", 0.25),
                    )
                else:
                    self.gate = Top2Gate(
                        self.args,
                        self.args.d_model,
                        self.args.moe_expert_count,
                        self.args.moe_gating_use_fp32,
                        self.args.moe_second_expert_policy,
                        self.args.moe_normalize_gate_prob_before_dropping,
                        getattr(self.args, "moe_eval_capacity_token_fraction", 0.25),
                        getattr(self.args, "moe_batch_prioritized_routing", False),
                    )
                self.experts = make_experts(self.args, self.hidden_size, self.intermediate_size, None)
                for p in self.experts.parameters():
                    p.expert = True  # type: ignore
                    
        self.num_local_experts = len(self.experts)
        print("gpu:{}, self.num_local_experts : {}".format(get_global_rank(), self.num_local_experts))
        self.in_generation = False
        self.a2a_cuda_event_intervals = []
        self.a2a_cpu_time_ms = 0.0
        self.reset_params = False

    def forward(self, *input: Tensor, input_padding_mask=None, train_meta_data=None, moe_idx=None, **kwargs: Any) -> Tensor:
        moe_start_time = time.time() * 1000
        meta_data = None
        if  self.args.gate_level == "sample":
            
            input = input[0]

            assert len(input.shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
            if input_padding_mask is not None:
                assert len(input_padding_mask.shape) == 2, "input Tensor must have dimensions: (s)equence, (t)oken"
                assert input_padding_mask.shape[0] == input.shape[0]
                assert input_padding_mask.shape[1] == input.shape[1]
            d_model = input.shape[2]
            l_aux, top_k_weights, top_k_indices, meta_data = self.gate(input, None, train_meta_data, moe_idx)
            
            B, S = input.shape[0], input.shape[1]
            
            if self.args.moe_gate_input_type == "stack":
                chunks = input.chunk(B, dim=0)
                top_k_weights_chunks = top_k_weights.chunk(B, dim=0)
                top_k_indices_chunks = top_k_indices.chunk(B, dim=0)

                results_all = []
                for chunk, weights, indices in zip(chunks,top_k_weights_chunks,top_k_indices_chunks):
                    res = []
                    
                    for indx in range(indices.size(-1) + self.args.moe_share_expert_count):
                        if indx < indices.size(-1):
                            exp = self.experts[indices[-1][indx].item()]
                            output = exp(chunk) * weights[-1][indx]
                        else:
                            exp = self.experts[self.args.moe_expert_count + indx - indices.size(-1)]
                            output = exp(chunk)
                        res.append(output)
                        
                    results_all.append(torch.stack(res,dim=-1).sum(-1).squeeze(0))


                experts_output = torch.stack(results_all, dim=0) # (B, S, M) 
            
            else:
                if self.args.max_sequence_length % self.args.gate_sample_ratio == 0:
                    B_sample = B * self.args.gate_sample_ratio
                else:
                    B_sample = B
                chunks = input.reshape(B_sample, int(S/int(B_sample/B)), d_model).chunk(B_sample, dim=0)
                top_k_weights_chunks = top_k_weights.chunk(B_sample, dim=0)
                top_k_indices_chunks = top_k_indices.chunk(B_sample, dim=0)
                
                results_all = []
                for chunk, weights, indices in zip(chunks,top_k_weights_chunks,top_k_indices_chunks):
                    res = []
    
                    for indx in range(indices.size(-1) + self.args.moe_share_expert_count):
                        if indx < indices.size(-1):
                            exp = self.experts[indices[-1][indx].item()]
                            output = exp(chunk) * weights[-1][indx]
                        else:
                            exp = self.experts[self.args.moe_expert_count + indx - indices.size(-1)]
                            output = exp(chunk)
                        res.append(output)
                        
                    results_all.append(torch.stack(res,dim=-1).sum(-1).squeeze(0))
                
                experts_output = torch.stack(results_all, dim=0).reshape(B, S, d_model) # (B, S, M) 
            
            return experts_output, l_aux, meta_data

        elif not self.args.gshard:
            input = input[0] # exclude meta data
            assert len(input.shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
            if input_padding_mask is not None:
                assert len(input_padding_mask.shape) == 2, "input Tensor must have dimensions: (s)equence, (t)oken"
                assert input_padding_mask.shape[0] == input.shape[0]
                assert input_padding_mask.shape[1] == input.shape[1]
            d_model = input.shape[2]
            # Pad to expected batch size
            input_shape = list(input.shape)

            # Reshape into S tokens by dropping sequence dimension.
            reshaped_input = input.reshape(-1, d_model)
            reshaped_input_shape = reshaped_input.shape
            reshaped_input_padding_mask = input_padding_mask.reshape(-1) if input_padding_mask is not None else None
                            
            l_aux, combine_weights, combine_mask, meta_data = self.gate(reshaped_input, reshaped_input_padding_mask)
            S, M = reshaped_input.size(0), reshaped_input.size(1)
            if self.args.moe_share_expert_count > 0:
                combine_weights = F.pad(combine_weights, (0, 0, 0, self.args.moe_share_expert_count), "constant", 1.0)

            results = []
            chunks = combine_weights.chunk(len(self.experts), dim=0)
            for model, chunk in zip(self.experts, chunks) :
                results.append(model(reshaped_input) * chunk.permute(1, 0))
            
            experts_output = torch.stack(results, dim=-1).sum(-1)   # (S, M) 
            combined_output = experts_output.reshape(input.shape)


            return combined_output, l_aux, meta_data

        else:
            input = input[0] # exclude meta data
            assert len(input.shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
            if input_padding_mask is not None:
                assert len(input_padding_mask.shape) == 2, "input Tensor must have dimensions: (s)equence, (t)oken"
                assert input_padding_mask.shape[0] == input.shape[0]
                assert input_padding_mask.shape[1] == input.shape[1]
            
            d_model = input.shape[2]
            # Pad to expected batch size
            input_shape = list(input.shape)
            expected_bsz = getattr(self.args, 'batch_size', 0) if self.training else getattr(self.args, 'batch_size_valid', 0)
            # This indicates that --batch-size or --max-sentences is not specified
            if expected_bsz is None:
                expected_bsz = 0
            # Note: Padding is not necessary at generation time at present
            # because all DDP workers process the same batch. Also, batch size at generation time
            # can be different from that present in the checkpoint state
            if not self.in_generation and expected_bsz != 0 and input_shape[0] != expected_bsz:
                log.warning(f"padding batch with unexpected size {input_shape[0]} (expected: {expected_bsz})")
                assert input_shape[0] < expected_bsz, f"{input_shape[0]} < {expected_bsz}"
                padded_input = torch.zeros(
                    (expected_bsz, input_shape[1], input_shape[2]),
                    dtype=input.dtype, layout=input.layout, device=input.device)
                padded_input[:input_shape[0], :, :] = input
                input = padded_input

                padded_input_padding_mask = torch.ones(
                    (expected_bsz, input_shape[1], ), dtype=torch.bool, device=input.device
                )
                if input_padding_mask is not None:
                    padded_input_padding_mask[:input_shape[0], :] = input_padding_mask
                else:
                    padded_input_padding_mask[:input_shape[0], :] = False
                input_padding_mask = padded_input_padding_mask

            # Reshape into S tokens by dropping sequence dimension.
            reshaped_input = input.reshape(-1, d_model)
            reshaped_input_shape = reshaped_input.shape
            reshaped_input_padding_mask = input_padding_mask.reshape(-1) if input_padding_mask is not None else None

            # Doing padding here when --max-tokens is specified and not --batch-size or --max-sentences
            # Pro of --max-tokens: more flexible for MT variable sequence lengths
            # Con of --max-tokens: extra all-reduce needed to figure out optimal padding without running OOM
            if expected_bsz == 0:
                expected_dim = int(all_reduce(
                    reshaped_input_shape[0] * torch.ones((1,), dtype=torch.long, device=input.device),
                    group=dist.group.WORLD,
                    op="max",
                ).item())
                padded_input = torch.zeros(
                    (expected_dim, reshaped_input_shape[1]),
                    dtype=input.dtype, layout=input.layout, device=input.device)
                padded_input[:reshaped_input_shape[0], :] = reshaped_input
                reshaped_input = padded_input

                padded_input_padding_mask = torch.ones(
                    (expected_dim,), dtype=torch.bool, device=padded_input.device
                )
                if reshaped_input_padding_mask is not None:
                    padded_input_padding_mask[:reshaped_input_shape[0]] = reshaped_input_padding_mask
                else:
                    padded_input_padding_mask[:reshaped_input_shape[0]] = False
                reshaped_input_padding_mask = padded_input_padding_mask

            
            l_aux, combine_weights, dispatch_mask = self.gate(reshaped_input, reshaped_input_padding_mask)
            dispatch_mask = dispatch_mask.to(input.dtype).permute(1, 2, 0)  # S,E,C -> E,C,S
            E, C, S = dispatch_mask.size()
            M = reshaped_input.size(1)
            assert reshaped_input.size() == (S, M)
            # einsum("sec,sm->ecm")
            dispatched_input = torch.mm(dispatch_mask.view(E*C, S), reshaped_input)  # -> (E*C),M

            if self.all2all_size > 1:
                dispatched_input = self.all_to_all_wrapper(dispatched_input)

            # Re-shape after all-to-all: ecm -> gecm
            dispatched_input = dispatched_input.reshape(self.all2all_size, self.num_local_experts, -1, d_model)
            chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
            expert_outputs = []
            for chunk, expert in zip(chunks, self.experts):
                expert_outputs += [expert(chunk)]
            expert_output = torch.cat(expert_outputs, dim=1)

            if self.all2all_size > 1:
                expert_output = self.all_to_all_wrapper(expert_output)

            # Re-shape back: gecm -> ecm
            expert_output = expert_output.reshape(self.all2all_size * self.num_local_experts, -1, d_model)
            
            combined_output = combine_weights.view(S, E*C).mm(expert_output.view(E*C, M))

            # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
            combined_output = combined_output[:reshaped_input_shape[0], :]
            combined_output = combined_output.reshape(input.shape)
            combined_output = combined_output[:input_shape[0], :, :]

            return combined_output, l_aux, meta_data



    def reset_parameters(self):
        if not self.reset_params:
            for expert in self.experts:
                expert.reset_parameters()
            self.gate.reset_parameters()
            
            self.reset_params = True

    def prepare_for_inference_(self):
        self.in_generation = True

    def all_to_all_wrapper(self, input: Tensor):
        dummy_a2a = getattr(self.args, 'dummy_a2a', False)
        if dummy_a2a:
            input = input.contiguous()
            output = input.detach().clone()
            return input
        output = _AllToAll.apply(self.all2all_group, input)
        return output