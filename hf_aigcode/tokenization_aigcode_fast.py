from transformers import AutoTokenizer, PreTrainedTokenizerFast

from hf_aigcode.configuration_aigcode import AIGCodeConfig


class AIGCodeTokenizerFast(PreTrainedTokenizerFast):
    # Note: AIGCode's tokenizer is already a wrapper around huggingface. This is potentially unnecessary.
    pass

AutoTokenizer.register(AIGCodeConfig, fast_tokenizer_class=AIGCodeTokenizerFast)
