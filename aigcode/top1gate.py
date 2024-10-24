from typing import Callable, Dict, Tuple, Optional

import math
import torch
from torch import Tensor
from torch.distributions import Categorical
from .config import (
    ModelConfig,
    InitFnType,
)

from .initialization import init_normal
import torch.nn.functional as F

gumbel_map: Dict[torch.device, Callable] = {}
fused_cumsum_sub_one = lambda mask: torch.cumsum(mask, dim=0) - 1

SAMPLE_FRACTION = 0.2


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


def one_hot(indices: torch.Tensor, num_classes: int, unsqueeze_indices=False) -> Tensor:
    if unsqueeze_indices:
        indices = indices.unsqueeze(-1)
    assert indices.shape[-1] == 1, "last dimension of indices must be have size 1"
    output = torch.zeros(indices.shape[:-1] + (num_classes,), device=indices.device, dtype=indices.dtype)
    output.scatter_(
        len(output.shape) - 1, indices, 1
    )
    return output


def kl_divergence(p, q):
    p = torch.clamp(p, min=1e-10)
    q = torch.clamp(q, min=1e-10)
    kl = torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)
    
    return kl


def entropy(probs):
    logits = torch.distributions.utils.probs_to_logits(probs)
    p_log_p = probs * logits
    return -p_log_p.sum(-1)

def top1gating_sample_level(
    config,
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    use_fp32=False,
    eval_mode=False,
    moe_eval_capacity_token_fraction=0.25,
    logging=False,
    gate_routing_labels=None,
    temperature=1.0,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Implements Top1Gating on logits."""
    metadata = {}
    if use_fp32:
        orig_dtype = logits.dtype
        logits = logits.float()

    gates = F.softmax(logits/temperature, dim=-1)
    if len(gates.shape) == 3:
        gates = gates.mean(1)

    if gate_routing_labels is not None:
        l_aux = torch.mean(kl_divergence(gate_routing_labels, gates))
    else:
        l_aux = None

    # gates has shape of B,E
    top_k_scores, top_k_indices = gates.topk(1, dim=-1) # B, K


    if logging:
        num_bs = gates.shape[0]
        num_experts = gates.shape[1]
        metadata["experts_top1_distribution"] = torch.histc((top_k_indices[:, 0] + 1), bins=num_experts, min=1, max=num_experts).float() / num_bs
        metadata["gate_score_distribution"] = torch.mean(gates, dim=0)
    return l_aux, top_k_scores, top_k_indices, metadata


def top1gating(
    config,
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    use_fp32=False,
    eval_mode=False,
    moe_eval_capacity_token_fraction=0.25,
    logging=False,
    temperature=1.0,
    layer_id=None,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Implements Top1Gating on logits."""
    metadata = {}
    if use_fp32:
        orig_dtype = logits.dtype
        logits = logits.float()
    gates = F.softmax(logits/temperature, dim=1)
    
    top1_probs, top1_indices = torch.topk(gates, k=1, dim=1)
    mask = torch.zeros_like(gates)
    combine_weights = mask.scatter_(1, top1_indices, top1_probs)
    combine_weights = combine_weights.permute(1, 0)

    if config.moe_auxiliary_loss:
    # Compute l_aux
        num_experts = gates.shape[1]
        indices1_s = torch.argmax(gates, dim=1, keepdim=True)
        mask1 = one_hot(indices1_s, num_experts)
        me = torch.mean(gates, dim=0)
        ce = torch.mean(mask1.to(gates.dtype), dim=0)
        l_aux = torch.mean(me * ce)
        l_aux = l_aux * num_experts * num_experts
    else:
        l_aux = None

    if logging:
        num_bs = gates.shape[0]
        num_experts = gates.shape[1]
        if layer_id:
            metadata["experts_top1_distribution_{}".format(layer_id)] = torch.histc((top1_indices[:, 0].to(torch.int32) + 1), bins=num_experts, min=1, max=num_experts).float() / num_bs
            metadata["gate_score_distribution_{}".format(layer_id)] = torch.mean(gates, dim=0)
        else:
            metadata["experts_top1_distribution"] = torch.histc((top1_indices[:, 0].to(torch.int32) + 1), bins=num_experts, min=1, max=num_experts).float() / num_bs
            metadata["gate_score_distribution"] = torch.mean(gates, dim=0)

    return l_aux, combine_weights, None, metadata



def top1gating_dist(
    config,
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    use_fp32=False,
    eval_mode=False,
    moe_eval_capacity_token_fraction=0.25,
    logging=False,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Implements Top1Gating on logits."""
    metadata = {}
    if use_fp32:
        orig_dtype = logits.dtype
        logits = logits.float()
    gates = F.softmax(logits, dim=1)
    
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    if moe_eval_capacity_token_fraction > 0.0 and eval_mode:
        capacity = math.ceil(moe_eval_capacity_token_fraction * num_tokens)
    else:
        capacity = 2 * math.ceil(num_tokens / num_experts)

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1, keepdim=True)
    mask1 = one_hot(indices1_s, num_experts)
    if second_expert_policy == 'sampling':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    else:
        logits_w_noise = logits
    
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    
    gates1_s = (gates * mask1).sum(dim=1)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.to(gates.dtype), dim=0)
    l_aux = torch.mean(me * ce)
    l_aux = l_aux * num_experts * num_experts
    mask1 = mask1 * torch.lt(locations1, capacity)
    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1.to(gates1_s.dtype)  # einsum("s,se->se")
    # gates2 = gates2_s.unsqueeze(-1) * mask2.to(gates2_s.dtype)  # einsum("s,se->se")
    locations1_sc = one_hot(locations1_s, num_classes=capacity, unsqueeze_indices=True)
    # locations2_sc = one_hot(locations2_s, num_classes=capacity, unsqueeze_indices=True)
    combine1_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates1.unsqueeze(-1), locations1_sc.to(gates1.dtype).unsqueeze(1)
    )

    combine_weights = combine1_sec 
    
    dispatch_mask = combine_weights.bool()
    if use_fp32:
        return l_aux, combine_weights.to(orig_dtype), dispatch_mask, metadata
    else:
        return l_aux, combine_weights, dispatch_mask, metadata


class Top1Gate(torch.nn.Module):
    """Gate module which implements Top1Gating as described in Gshard_.
    ::

        gate = Top1Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        config: ModelConfig,
        model_dim: int,
        num_experts: int,
        layer_id: int,
        use_fp32=False,
        moe_eval_capacity_token_fraction=0.25,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = model_dim
        self.num_experts = num_experts
        if self.config.gate_level == "sample":
            self.max_sequence_length = self.config.max_sequence_length
            if self.config.moe_gate_input_type == "stack":
                self.wg = torch.nn.Linear(int(model_dim/self.config.gate_sample_ratio), num_experts, bias=False)
                self.aux_wg = torch.nn.Linear(model_dim, int(model_dim/self.config.gate_sample_ratio), bias=False)
            else:
                if self.max_sequence_length % self.config.gate_sample_ratio == 0:
                    self.sample_tokens_length = int(self.max_sequence_length / self.config.gate_sample_ratio)
                else:
                    self.sample_tokens_length = self.max_sequence_length
                    
                self.wg = torch.nn.Linear(num_experts * self.sample_tokens_length, num_experts, bias=False)
                self.aux_wg = torch.nn.Linear(model_dim, num_experts, bias=False)

        else:
            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.use_fp32 = use_fp32
        
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction
        self.reset_params = False
        
    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor]=None, train_meta_data: Optional[Dict]=None, moe_idx: Optional[int]=None) -> Tuple[Tensor, Tensor, Tensor, Dict]:  # type: ignore
        
        if self.config.gate_level == "sample":
            if self.config.moe_gate_no_grad:
                with torch.no_grad():
                    x = self.aux_wg(input)
                    x_shape = list(x.shape)

                    if train_meta_data is not None and "rand_mask_indices" in train_meta_data:
                    
                        xs = x.chunk(x_shape[0], dim=0)
                        reshape_input_res = []
                        for x_i, rand_mask_idx in zip(xs, train_meta_data["rand_mask_indices"]):
                            x_i = x_i[:, :rand_mask_idx, :]
                            
                            if x_i.size(1) < self.max_sequence_length:
                                repeat_times = self.max_sequence_length // x_i.size(1) + 1
                                reshape_input_i = x_i.repeat(1, repeat_times, 1)
                                reshape_input_i = reshape_input_i[:, :self.max_sequence_length, :].reshape(self.max_sequence_length * x_shape[2])
                            elif x_i.size(1) > self.max_sequence_length:
                                reshape_input_i = x_i[:, :self.max_sequence_length, :].reshape(self.max_sequence_length * x_shape[2])
                            else:
                                reshape_input_i = x_i.reshape(self.max_sequence_length * x_shape[2])
                            reshape_input_res.append(reshape_input_i)
                        reshape_input = torch.stack(reshape_input_res, dim=0)
                       
                    else:
                        if x_shape[1] < self.max_sequence_length:
                            repeat_times = self.max_sequence_length // x_shape[1] + 1
                            reshape_input = x.repeat(1, repeat_times, 1)
                            reshape_input = reshape_input[:, :self.max_sequence_length, :].reshape(x_shape[0], self.max_sequence_length * x_shape[2])
                        elif x_shape[1] > self.max_sequence_length:
                            reshape_input = x[:, :self.max_sequence_length, :].reshape(x_shape[0], self.max_sequence_length * x_shape[2])
                        else:
                            reshape_input = x.reshape(x_shape[0], x_shape[1] * x_shape[2])
                    

                    logits = self.wg(reshape_input)
            else:
                x = self.aux_wg(input)
                x_shape = list(x.shape)

                if train_meta_data is not None and "rand_mask_indices" in train_meta_data:
                    
                    xs = x.chunk(x_shape[0], dim=0)
                    reshape_input_res = []
                    for x_i, rand_mask_idx in zip(xs, train_meta_data["rand_mask_indices"]):
                        x_i = x_i[:, :rand_mask_idx, :]
                        
                        if x_i.size(1) < self.max_sequence_length:
                            repeat_times = self.max_sequence_length // x_i.size(1) + 1
                            reshape_input_i = x_i.repeat(1, repeat_times, 1)
                            reshape_input_i = reshape_input_i[:, :self.max_sequence_length, :].reshape(self.max_sequence_length * x_shape[2])
                        elif x_i.size(1) > self.max_sequence_length:
                            reshape_input_i = x_i[:, :self.max_sequence_length, :].reshape(self.max_sequence_length * x_shape[2])
                        else:
                            reshape_input_i = x_i.reshape(self.max_sequence_length * x_shape[2])
                        reshape_input_res.append(reshape_input_i)
                    reshape_input = torch.stack(reshape_input_res, dim=0)

                else:
                    if self.config.moe_gate_input_type == "stack":
                        logits = self.wg(self.aux_wg(input))

                    else:
                        ### sample tokens
                        x = x.reshape(int(x_shape[0] * (self.max_sequence_length / self.sample_tokens_length)),  self.sample_tokens_length, x_shape[2])
                        x_shape_ = list(x.shape)
                        if x_shape_[1] < self.sample_tokens_length:
                            # reshape_input = F.pad(reshape_input, (0, self.num_experts * self.max_sequence_length - reshape_input.size(-1), 0, 0), mode='constant', value=0.)
                            repeat_times = self.sample_tokens_length // x_shape_[1] + 1
                            reshape_input = x.repeat(1, repeat_times, 1)
                            reshape_input = reshape_input[:, :self.sample_tokens_length, :].reshape(x_shape_[0], self.sample_tokens_length * x_shape_[2])
                        elif x_shape_[1] > self.sample_tokens_length:
                            reshape_input = x[:, :self.sample_tokens_length, :].reshape(x_shape_[0], self.sample_tokens_length * x_shape_[2])
                        else:
                            reshape_input = x.reshape(x_shape_[0], x_shape_[1] * x_shape_[2])

                        logits = self.wg(reshape_input)
                
            if train_meta_data is not None and "routing_labels" in train_meta_data:
                gate_routing_labels = torch.tensor([gate_label[moe_idx] for gate_label in train_meta_data["routing_labels"]], device=logits.device)
            else:
                gate_routing_labels = None

            return top1gating_sample_level(
                self.config,
                logits,
                mask,
                use_fp32=self.use_fp32,
                eval_mode=not self.training,
                moe_eval_capacity_token_fraction=self.moe_eval_capacity_token_fraction,
                logging=self.config.moe_logging,
                gate_routing_labels=gate_routing_labels,
                temperature=self.config.gate_softmax_temperature,
            )
        else:
            logits = self.wg(input)
            if not self.config.gshard:
                return top1gating(
                    self.config,
                    logits,
                    mask,
                    use_fp32=self.use_fp32,
                    eval_mode=not self.training,
                    moe_eval_capacity_token_fraction=self.moe_eval_capacity_token_fraction,
                    logging=self.config.moe_logging,
                    temperature=self.config.gate_softmax_temperature,
                    layer_id=self.layer_id,
                )
            else:
                return top1gating_dist(
                self.config,
                logits,
                mask,
                use_fp32=self.use_fp32,
                eval_mode=not self.training,
                moe_eval_capacity_token_fraction=self.moe_eval_capacity_token_fraction,
                logging=False,
                )

    def reset_parameters(self):
            if not self.reset_params:
                if self.config.gate_level == "sample":
                    if self.config.init_fn == InitFnType.normal:
                        std = self.config.init_std
                        cutoff_factor = self.config.init_cutoff_factor
                    elif self.config.init_fn == InitFnType.mitchell:
                        std = 1 / math.sqrt(self.config.d_model)
                        std_ = 1 / math.sqrt(self.config.max_sequence_length * self.num_experts)
                        cutoff_factor = self.config.init_cutoff_factor or 3.0
                    elif self.config.init_fn == InitFnType.full_megatron:
                        std = self.config.init_std
                        cutoff_factor = self.config.init_cutoff_factor or 3.0
                    else:
                        raise NotImplementedError(self.config.init_fn)

                    init_normal(self.aux_wg, std, cutoff_factor)
                    init_normal(self.wg, std_, cutoff_factor)
                else:
                    if self.config.init_fn == InitFnType.normal:
                        std = self.config.init_std
                        cutoff_factor = self.config.init_cutoff_factor
                    elif self.config.init_fn == InitFnType.mitchell:
                        std = 1 / math.sqrt(self.config.d_model)
                        cutoff_factor = self.config.init_cutoff_factor or 3.0
                    elif self.config.init_fn == InitFnType.full_megatron:
                        std = self.config.init_std
                        cutoff_factor = self.config.init_cutoff_factor or 3.0
                    else:
                        raise NotImplementedError(self.config.init_fn)

                    init_normal(self.wg, std, cutoff_factor)
                    
                self.reset_params = True
