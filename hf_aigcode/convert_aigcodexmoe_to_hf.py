import argparse
import logging
import os
import shutil

import torch
from omegaconf import OmegaConf as om

from hf_aigcode.configuration_aigcodexmoe import AIGCcodeXMoEConfig
# from .configuration_aigcodexmoe import AIGCcodeXMoEConfig
from hf_aigcode.modeling_aigcodexmoe import AIGCcodeXMoEForCausalLM
# from .modeling_aigcodexmoe import AIGCcodeXMoEForCausalLM

# from hf_aigcodexmoe.tokenization_aigcode_fast import AIGCcodeTokenizerFast
from aigcode import ModelConfig, Tokenizer

logger = logging.getLogger(__name__)


def write_config(checkpoint_dir: str, to_hf_dir: str):
    # save config as HF config

    logger.info(f"Loading checkpoint from {checkpoint_dir}")

    config_path = os.path.join(checkpoint_dir, "config.yaml")
    model_config = ModelConfig.load(config_path, key="model")
    config_kwargs = model_config.asdict()
    config_kwargs["use_cache"] = True
    config = AIGCcodeXMoEConfig(**config_kwargs)

    logger.info(f"Saving HF-compatible config to {os.path.join(checkpoint_dir, 'config.json')}")
    config.save_pretrained(to_hf_dir)


def write_model(checkpoint_dir: str, to_hf_dir: str, ignore_aigcode_compatibility: bool = False):
    # For device_map = "auto", etc. the models are loaded in a way that start_prefix is not computed correctly.
    # So, we explicitly store the model with the expected prefix.

        
    old_model_path = os.path.join(checkpoint_dir, "model.pt")
    new_model_path = os.path.join(to_hf_dir, "pytorch_model.bin")

    state_dict = torch.load(old_model_path)
    new_state_dict = {f"{AIGCcodeXMoEForCausalLM.base_model_prefix}.{key}": val for key, val in state_dict.items()}
    torch.save(new_state_dict, new_model_path)

    if ignore_aigcode_compatibility:
        os.remove(old_model_path)


def write_tokenizer(checkpoint_dir: str):
    tokenizer_raw = Tokenizer.from_checkpoint(checkpoint_dir)
    tokenizer = AIGCcodeXMoETokenizerFast(
        tokenizer_object=tokenizer_raw.base_tokenizer,
        truncation=tokenizer_raw.truncate_direction,
        max_length=tokenizer_raw.truncate_to,
        eos_token=tokenizer_raw.decode([tokenizer_raw.eos_token_id], skip_special_tokens=False),
    )
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.pad_token_id = tokenizer_raw.pad_token_id
    tokenizer.eos_token_id = tokenizer_raw.eos_token_id

    tokenizer.save_pretrained(checkpoint_dir)


def convert_checkpoint(checkpoint_dir: str, to_hf_dir: str, ignore_aigcode_compatibility: bool = False):
    if not os.path.exists(to_hf_dir):
        os.makedirs(to_hf_dir)
    write_config(checkpoint_dir, to_hf_dir)
    write_model(checkpoint_dir, to_hf_dir, ignore_aigcode_compatibility=ignore_aigcode_compatibility)
    # write_tokenizer(checkpoint_dir)

    # Cannot remove it before writing the tokenizer
    if ignore_aigcode_compatibility:
        os.remove(os.path.join(checkpoint_dir, "config.yaml"))


def download_remote_checkpoint_and_convert_to_hf(checkpoint_dir: str, local_dir: str):
    from cached_path import cached_path

    model_name = os.path.basename(checkpoint_dir)
    local_model_path = os.path.join(local_dir, model_name)
    os.makedirs(local_model_path, exist_ok=True)

    model_files = ["model.pt", "config.yaml"]  # , "optim.pt", "other.pt"]
    for filename in model_files:
        final_location = os.path.join(local_model_path, filename)
        if not os.path.exists(final_location):
            remote_file = os.path.join(checkpoint_dir, filename)
            logger.debug(f"Downloading file {filename}")
            cached_file = cached_path(remote_file)
            shutil.copy(cached_file, final_location)
            logger.debug(f"File at {final_location}")
        else:
            logger.info(f"File already present at {final_location}")

    convert_checkpoint(local_model_path)
    return local_model_path


# def fix_bad_tokenizer(checkpoint_dir: str):
#     path = os.path.join(checkpoint_dir, "config.yaml")
#     conf = om.load(path)
#     conf["tokenizer"]["identifier"] = "allenai/gpt-neox-aigcode-dolma-v1_5"
#     conf["model"]["eos_token_id"] = 50279
#     om.save(conf, path)


def main():
    parser = argparse.ArgumentParser(
        description="Adds a config.json to the checkpoint directory, and creates pytorch_model.bin, "
        "making it easier to load weights as HF models."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/sharedata/aigcode_7b_checkpoints/814/step252000-unsharded",
        help="Location of AIGCcode checkpoint.",
    )
    parser.add_argument(
        "--to-hf-dir",
        default="/sharedata/zimoliu/models/aigcodexmoe_252000",
        help="output dir for hf models"
    )
    parser.add_argument(
        "--ignore-aigcode-compatibility",
        action="store_true",
        help="Ignore compatibility with the aigcode codebase. "
        "This will remove files that are needed specifically for aigcode codebase, eg. config.yaml, etc.",
    )

    args = parser.parse_args()
    print("args:{}\n".format(args))
    os.makedirs(args.to_hf_dir, exist_ok=True)
    # fix_bad_tokenizer(args.checkpoint_dir)
    convert_checkpoint(args.checkpoint_dir, args.to_hf_dir, args.ignore_aigcode_compatibility)


if __name__ == "__main__":
    main()
