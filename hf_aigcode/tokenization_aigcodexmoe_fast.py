from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .configuration_aigcodexmoe import AIGCcodeXMoEConfig

# to be done
class AIGCcodeXMoETokenizerFast(PreTrainedTokenizerFast):
    # Note: AIGCcode's tokenizer is already a wrapper around huggingface. This is potentially unnecessary.
    pass

