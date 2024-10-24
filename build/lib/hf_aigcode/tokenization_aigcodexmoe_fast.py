from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .configuration_olmoxmoe import OLMoXMoEConfig

# to be done
class OLMoXMoETokenizerFast(PreTrainedTokenizerFast):
    # Note: OLMo's tokenizer is already a wrapper around huggingface. This is potentially unnecessary.
    pass

    # def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
    #     # This is required to make the implementation complete.
    #     pass


# Register the tokenizer class so that it is available for transformer pipelines, auto-loading etc.
# AutoTokenizer.register(OLMoXMoEConfig, fast_tokenizer_class=OLMoXMoETokenizerFast)
