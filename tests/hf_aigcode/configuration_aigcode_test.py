import tempfile

from aigcode.config import ModelConfig


def test_config_save(model_path: str):
    from hf_aigcode.configuration_aigcode import AIGCcodeConfig

    with tempfile.TemporaryDirectory() as temp_dir:
        config = ModelConfig(alibi=True)  # default is False
        hf_config = AIGCcodeConfig(**config.asdict())

        hf_config.save_pretrained(temp_dir)
        loaded_hf_config = AIGCcodeConfig.from_pretrained(temp_dir)

        assert hf_config.to_dict() == loaded_hf_config.to_dict()

        for key, val in config.asdict().items():
            assert getattr(loaded_hf_config, key) == val
