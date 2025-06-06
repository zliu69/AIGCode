from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from aigcode import BlockType, LayerNorm, AIGCcode, Tokenizer, TrainConfig
from aigcode.config import ModelConfig, PaddingDirection
from aigcode.data import DataCollator


@pytest.mark.parametrize(
    "alibi, rope, flash_attn, block_type, multi_query_attention, cuda, dtype",
    [
        pytest.param(
            True, False, False, BlockType.sequential, False, False, torch.bfloat16, id="alibi-emb-cpu-bf16"
        ),
        pytest.param(
            False, False, False, BlockType.sequential, False, False, torch.bfloat16, id="abs-emb-cpu-bf16"
        ),
        pytest.param(
            True, False, False, BlockType.sequential, False, False, torch.float32, id="alibi-emb-cpu-f32"
        ),
        pytest.param(False, False, False, BlockType.sequential, False, False, torch.float32, id="abs-emb-cpu-f32"),
        pytest.param(
            False, True, False, BlockType.sequential, False, False, torch.bfloat16, id="rope-emb-cpu-bf16"
        ),
        pytest.param(False, True, False, BlockType.sequential, False, False, torch.float32, id="rope-emb-cpu-f32"),
        pytest.param(
            True,
            False,
            False,
            BlockType.sequential,
            False,
            True,
            torch.bfloat16,
            id="alibi-emb-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            False,
            True,
            False,
            BlockType.sequential,
            False,
            True,
            torch.bfloat16,
            id="rope-emb-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            False,
            False,
            False,
            BlockType.sequential,
            False,
            True,
            torch.bfloat16,
            id="abs-emb-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            False,
            False,
            True,
            BlockType.sequential,
            False,
            True,
            torch.bfloat16,
            id="abs-emb-flash-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            False,
            False,
            True,
            BlockType.sequential,
            False,
            True,
            torch.float16,
            id="abs-emb-flash-cuda-f16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            False, False, False, BlockType.sequential, True, False, torch.float32, id="abs-emb-mqattn-cpu-f32"
        ),
        pytest.param(
            True,
            False,
            False,
            BlockType.sequential,
            True,
            False,
            torch.float32,
            id="alibi-emb-mqattn-cpu-f32",
        ),
    ],
)
def test_forward(
    train_config: TrainConfig,
    tokenizer: Tokenizer,
    alibi: bool,
    rope: bool,
    flash_attn: bool,
    block_type: BlockType,
    multi_query_attention: bool,
    cuda: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if cuda else "cpu")

    train_config.model.alibi = alibi
    train_config.model.rope = rope
    train_config.model.flash_attention = flash_attn
    if flash_attn:
        train_config.model.attention_dropout = 0.0
    train_config.model.block_type = block_type
    train_config.model.multi_query_attention = multi_query_attention
    if cuda:
        train_config.model.init_device = "cuda"
    else:
        train_config.model.init_device = "cpu"

    use_amp = dtype in {torch.float16, torch.bfloat16}

    model = AIGCcode(train_config.model).eval()

    input1 = tokenizer.encode("My name is AIGCcode!")
    input2 = tokenizer.encode("I'm a delightful large open language model :)")
    batch_inputs = DataCollator.from_train_config(train_config)(
        [  # type: ignore
            {"input_ids": input1, "attention_mask": [1.0] * len(input1)},
            {"input_ids": input2, "attention_mask": [1.0] * len(input2)},
        ]
    )
    batch_inputs = {  # type: ignore
        k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()
    }

    # Run forward pass.
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda" if cuda else "cpu", enabled=use_amp, dtype=None if not use_amp else dtype
        ):
            output1 = model(torch.tensor(input1, device=device).unsqueeze(0))
            key_value_cache1 = model(
                torch.tensor(input1[:-1], device=device).unsqueeze(0), use_cache=True
            ).attn_key_values
            output1_from_cached = model(
                torch.tensor(input1[-1:], device=device).unsqueeze(0), past_key_values=key_value_cache1
            )
            output2 = model(torch.tensor(input2, device=device).unsqueeze(0))
            batch_output = model(**batch_inputs)
            batch_key_value_cache = model(
                batch_inputs["input_ids"][:, :-1],
                attention_mask=batch_inputs["attention_mask"][:, :-1],
                use_cache=True,
            ).attn_key_values
            batch_output_from_cached = model(
                batch_inputs["input_ids"][:, -1].unsqueeze(1),
                attention_mask=batch_inputs["attention_mask"],
                past_key_values=batch_key_value_cache,
            )

    # With using half-precision types these might have some big differences in a small
    # percentage of the elements.
    atol = 1e-2 if use_amp else None
    rtol = 1e3 if use_amp else None

    # Check that logits from individual inputs are equal to logits from batch.
    torch.testing.assert_close(
        output1.logits[0][: len(input1)], batch_output.logits[0][: len(input1)], rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        output2.logits[0][: len(input2)], batch_output.logits[1][: len(input2)], rtol=rtol, atol=atol
    )

    # Check that output using cached attention keys + values matches.
    torch.testing.assert_close(output1.logits[0][-1], output1_from_cached.logits[0][-1], rtol=rtol, atol=atol)
    # For the batched output this only makes sense for the longer of the two inputs, since the shorter one is padded on the right.
    torch.testing.assert_close(output2.logits[0][-1], batch_output_from_cached.logits[1][-1], rtol=rtol, atol=atol)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device")
@pytest.mark.parametrize("positional_embeddings", (None, "rope"))
@pytest.mark.parametrize("block_type", (BlockType.sequential,))
@pytest.mark.parametrize("attention_mode", ("mha", "mqa", "gqa"))
def test_flash_attn(
    train_config: TrainConfig,
    tokenizer: Tokenizer,
    positional_embeddings: Optional[str],
    block_type: BlockType,
    attention_mode: str,
):
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")

    if positional_embeddings is None:
        alibi = False
        rope = False
    elif positional_embeddings == "alibi":
        alibi = True
        rope = False
    elif positional_embeddings == "rope":
        alibi = False
        rope = True
    else:
        raise ValueError(f"{positional_embeddings} is not a valid value for positional_embeddings")

    def make_model(flash_attn: bool):
        torch.manual_seed(0)
        train_config.model.alibi = alibi
        train_config.model.rope = rope
        train_config.model.flash_attention = flash_attn
        train_config.model.attention_dropout = 0.0
        train_config.model.block_type = block_type
        train_config.model.d_model = 1024
        train_config.model.n_heads = 8
        if attention_mode == "mha":
            train_config.model.n_kv_heads = train_config.model.n_heads
        elif attention_mode == "mqa":
            train_config.model.n_kv_heads = 1
        elif attention_mode == "gqa":
            train_config.model.n_kv_heads = train_config.model.n_heads // 2
        else:
            raise ValueError(f"{attention_mode} is not a valid value for attention_mode")
        train_config.model.init_device = "cuda"
        return AIGCcode(train_config.model).eval()

    input1 = tokenizer.encode(
        "As a large language model, I don’t have personal opinions, but I can share some interesting facts!"
    )
    input2 = tokenizer.encode("What do you call a programmer with no bugs in their code? A liar.")
    input3 = tokenizer.encode("How do you comfort a JavaScript bug? You console it.")
    batch_inputs = DataCollator.from_train_config(train_config)(
        [  # type: ignore
            {"input_ids": input1},
            {"input_ids": input2},
            {"input_ids": input3},
        ]
    )
    batch_inputs = {  # type: ignore
        k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()
    }

    model_with_flash = make_model(True)
    model_without_flash = make_model(False)

    # Run forward pass.
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output_with_flash = model_with_flash(**batch_inputs)
            output_without_flash = model_without_flash(**batch_inputs)

    # With using half-precision types these might have some big differences in a small
    # percentage of the elements.
    atol = 1e-2
    rtol = 1e3

    # Check that logits match
    torch.testing.assert_close(output_with_flash.logits, output_without_flash.logits, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "alibi, flash_attn, cuda, dtype",
    [
        pytest.param(True, False, False, torch.bfloat16, id="alibi-emb-cpu-bf16"),
        pytest.param(False, False, False, torch.bfloat16, id="abs-emb-cpu-bf16"),
        pytest.param(
            True,
            False,
            True,
            torch.bfloat16,
            id="alibi-emb-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            False,
            False,
            True,
            torch.bfloat16,
            id="abs-emb-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        pytest.param(
            False,
            True,
            True,
            torch.bfloat16,
            id="abs-emb-flash-cuda-bf16",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
                pytest.mark.skipif(
                    torch.cuda.device_count() < 1 or "A100" not in torch.cuda.get_device_name(),
                    reason="Requires A100 GPU type",
                ),
            ),
        ),
    ],
)
def test_backward(
    train_config: TrainConfig, tokenizer: Tokenizer, alibi: bool, flash_attn: bool, cuda: bool, dtype
):
    torch.manual_seed(0)
    device = torch.device("cuda" if cuda else "cpu")

    use_amp = dtype in {torch.float16, torch.bfloat16}
    scaler = None if not (cuda and use_amp) else torch.cuda.amp.GradScaler()

    train_config.model.alibi = alibi
    train_config.model.flash_attention = flash_attn
    if flash_attn:
        train_config.model.attention_dropout = 0.0
    if cuda:
        train_config.model.init_device = "cuda"
    else:
        train_config.model.init_device = "cpu"

    model = AIGCcode(train_config.model).train()

    with torch.autocast(
        device_type="cuda" if cuda else "cpu", enabled=use_amp, dtype=None if not use_amp else dtype
    ):
        # Forward pass to get logits.
        input_ids = torch.tensor(tokenizer.encode("My name is AIGCcode!"), device=device).unsqueeze(0)
        logits = model(input_ids).logits

        # Compute loss.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Backward pass.
    if scaler is not None:
        scaler.scale(loss).backward()  # type: ignore
    else:
        loss.backward()

    # Check gradients.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            zeros = torch.zeros(parameter.size(), device=device)
            if (parameter.grad == zeros).all():
                raise RuntimeError(f"{name} has zero a gradient!")
        else:
            assert parameter.grad is None


@pytest.mark.parametrize(
    "cuda, dtype",
    [
        pytest.param(False, torch.float32, id="cpu-fp32"),
        pytest.param(
            True,
            torch.float32,
            id="cuda-fp32",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device"),
            ),
        ),
        # TODO: with an uninitialized model like we have here we'll end up with nan's
        # when we use half-precision. So eventually we should use a trained model in these tests.
        #  pytest.param(False, torch.bfloat16, id="cpu-bf16"),
    ],
)
def test_generate(
    train_config: TrainConfig,
    tokenizer: Tokenizer,
    cuda: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if cuda else "cpu")

    # Should always pad left when generating.
    train_config.data.pad_direction = PaddingDirection.left
    # We also need to use a relative positional embedding so that the
    # padding doesn't affect the results.
    train_config.model.alibi = True

    if cuda:
        train_config.model.init_device = "cuda"
    else:
        train_config.model.init_device = "cpu"
    use_amp = dtype in {torch.float16, torch.bfloat16}

    model = AIGCcode(train_config.model).eval()

    input1 = tokenizer.encode("My name is AIGCcode! ", add_special_tokens=False)
    input2 = tokenizer.encode("I'm a delightful large open language model :) ", add_special_tokens=False)
    batch_inputs = DataCollator.from_train_config(train_config)(
        [  # type: ignore
            {"input_ids": input1, "attention_mask": [1.0] * len(input1)},
            {"input_ids": input2, "attention_mask": [1.0] * len(input2)},
        ]
    )
    batch_inputs = {  # type: ignore
        k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()
    }
    beam_search_kwargs = dict(beam_size=3, max_steps=5)

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda" if cuda else "cpu", enabled=use_amp, dtype=None if not use_amp else dtype
        ):
            output1 = model.generate(
                torch.tensor(input1, device=device).unsqueeze(0),  # type: ignore
                **beam_search_kwargs,
            )
            batch_output = model.generate(**{**batch_inputs, **beam_search_kwargs})

    torch.testing.assert_close(output1.scores[0], batch_output.scores[0])


@pytest.mark.parametrize("elementwise_affine", (True, False))
@pytest.mark.parametrize("include_bias", (True, False))
def test_layer_norm(train_config: TrainConfig, elementwise_affine: bool, include_bias: bool):
    train_config.model.layer_norm_with_affine = elementwise_affine
    train_config.model.include_bias = include_bias
    ln = LayerNorm.build(train_config.model)

    needs_weight = elementwise_affine
    needs_bias = elementwise_affine and include_bias
    with torch.no_grad():
        if needs_weight:
            weight = torch.randn(train_config.model.d_model)
            ln.weight.copy_(weight)
        else:
            weight = None

        if needs_bias:
            bias = torch.randn(train_config.model.d_model)
            ln.bias.copy_(bias)
        else:
            bias = None

    assert ln.bias is None or ln.bias.requires_grad == needs_bias
    assert ln.weight is None or ln.weight.requires_grad == needs_weight

    x = torch.randn(16, 1024, train_config.model.d_model)
    x.requires_grad = False
    y_expected = F.layer_norm(x, [train_config.model.d_model], weight, bias)

    y_actual = ln(x)
    torch.testing.assert_close(y_actual, y_expected)


def test_block_groups():
    model_with_block_groups = AIGCcode(ModelConfig(d_model=128, n_heads=2, n_layers=9, block_group_size=3)).eval()
    model_without_block_groups = AIGCcode(ModelConfig(d_model=128, n_heads=2, n_layers=9, block_group_size=1)).eval()

    # We should be able to load the state dict from one model into the other, and vice-versa.
    state_dict_to_load, og_keys_to_new_keys = model_with_block_groups._make_state_dict_compatible(
        model_without_block_groups.state_dict()
    )
    assert og_keys_to_new_keys["transformer.blocks.0.attn_out.weight"] == {
        "transformer.block_groups.0.0.attn_out.weight"
    }
    model_with_block_groups.load_state_dict(state_dict_to_load)

    state_dict_to_load, og_keys_to_new_keys = model_without_block_groups._make_state_dict_compatible(
        model_with_block_groups.state_dict()
    )
    assert og_keys_to_new_keys["transformer.block_groups.0.0.attn_out.weight"] == {
        "transformer.blocks.0.attn_out.weight"
    }
    model_without_block_groups.load_state_dict(state_dict_to_load)

    # Check that output is exactly the same.
    input_ids = torch.randint(0, model_with_block_groups.config.vocab_size, (2, 16))
    with torch.no_grad():
        block_groups_output = model_with_block_groups(input_ids)
        no_block_groups_output = model_without_block_groups(input_ids)

    torch.testing.assert_close(block_groups_output, no_block_groups_output)
