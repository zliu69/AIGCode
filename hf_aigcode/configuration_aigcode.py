"""
AIGCode configuration
"""

from transformers import AutoConfig, PretrainedConfig
from transformers.utils import logging

from aigcode.config import ModelConfig

logger = logging.get_logger(__name__)


class AIGCodeConfig(PretrainedConfig):
    model_type = "hf_aigcode"
    keys_to_ignore_at_inference = ["past_key_values"]  # TODO: confirm

    def __init__(self, use_cache: bool = False, **kwargs):
        model_config = ModelConfig()
        all_kwargs = model_config.asdict()
        all_kwargs.update(kwargs)
        all_kwargs.update({"use_cache": use_cache})
        all_kwargs.update(
            {"architectures": all_kwargs.get("architectures", ["AIGCodeForCausalLM"]) or ["AIGCodeForCausalLM"]}
        )
        super().__init__(**all_kwargs)

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers

    @property
    def hidden_size(self):
        return self.d_model

AutoConfig.register("hf_aigcode", AIGCodeConfig)
