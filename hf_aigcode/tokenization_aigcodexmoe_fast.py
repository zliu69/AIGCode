from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .configuration_aigcodexmoe import AIGCodeXMoEConfig

# to be done
class AIGCodeXMoETokenizerFast(PreTrainedTokenizerFast):
    # Note: AIGCode's tokenizer is already a wrapper around huggingface. This is potentially unnecessary.
    pass

