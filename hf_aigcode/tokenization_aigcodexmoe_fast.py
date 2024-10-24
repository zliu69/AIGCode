from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .configuration_aigcodexmoe import AIGCcodeXMoEConfig

# to be done
class AIGCcodeXMoETokenizerFast(PreTrainedTokenizerFast):
    # Note: AIGCcode's tokenizer is already a wrapper around huggingface. This is potentially unnecessary.
    pass

    # def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
    #     # This is required to make the implementation complete.
    #     pass


# Register the tokenizer class so that it is available for transformer pipelines, auto-loading etc.
# AutoTokenizer.register(AIGCcodeXMoEConfig, fast_tokenizer_class=AIGCcodeXMoETokenizerFast)
