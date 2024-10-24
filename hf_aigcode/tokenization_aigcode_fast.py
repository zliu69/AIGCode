from transformers import AutoTokenizer, PreTrainedTokenizerFast

from hf_aigcode.configuration_aigcode import AIGCcodeConfig


class AIGCcodeTokenizerFast(PreTrainedTokenizerFast):
    # Note: AIGCcode's tokenizer is already a wrapper around huggingface. This is potentially unnecessary.
    pass

AutoTokenizer.register(AIGCcodeConfig, fast_tokenizer_class=AIGCcodeTokenizerFast)
