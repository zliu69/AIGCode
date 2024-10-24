"""Run this script with 'python'."""

# import gzip

import os
import sys
import random
import numpy as np
from pathlib import Path
# from typing import Optional, TextIO

from aigcode.config import CheckpointType, TrainConfig
# from aigcode.data import build_train_dataloader
# from aigcode.eval import build_evaluators
from aigcode.exceptions import AIGCcodeCliError, AIGCcodeConfigurationError
# from aigcode.model import Aigcode

# from aigcode.train import Trainer
# from aigcode.train_eval import TrainEvaluator
# from aigcode.train_sft import SFTTrainer
# from aigcode.util import clean_opt, log_extra_field, prepare_cli_environment, _get_s3_client, file_size, get_bytes_range
from aigcode.tokenizer import Tokenizer
# import aigcode
# print(aigcode.__file__)
# print(aigcode.__path__)
# log = logging.getLogger("train")

# """python ./scripts/check_npy.py ./tokenizers/ver_0702_65K.json /sharedata/pretrain/npy/matrix/instruction/"""
# """python ./scripts/check_npy.py ./tokenizers/ver_0708_65K.json /sharedata/pretrain/npy/eval/v2_small_gptneox20b/mc4_en/"""

def main(cfg, npy_path) -> None:

    # Load tokenizer
    # tokenizer = Tokenizer.from_file(cfg)
    tokenizer = Tokenizer.from_file(cfg, eos_token_id=2, pad_token_id=0)
    np.set_printoptions(threshold=np.inf)
    # train_config = cfg
    # data_config = train_config.data
    # max_tokens = 256 * 1024 * 1024
        # max_tokens = 256 * 1024 * 1024
    max_tokens = 256 * 1024 * 4
    # max_size = 256 * 1024 * 1024 * np.dtype(uint32).item_size
        # for dir in data_config.data_dir:
    print("dir_root: {}\n".format(npy_path))
    for root, dirs, files in os.walk(npy_path):
        r_memmap_mask_array = None
        cnt = 0
        for f in files:
            if cnt >= 10:
                break
            else:
                cnt = cnt + 1
            if f.endswith(".npy"):
                fpath = str(os.path.join(root, f)) 
                print("="*20)
                print("\n\n fpath: {}\n".format(fpath))
                if "input_ids" in f:
                    r_memmap_array = np.memmap(filename=fpath, dtype=np.uint16, mode="r")
                    print("dir: {}\n".format(fpath))
                    print("r_memmap sample0 : {}\n".format(tokenizer.decode(r_memmap_array[4096:8192])))
                    # r_memmap_array = np.memmap(filename=fpath, dtype=np.uint16, mode="r")
                    print("r_memmap sample1 : {}\n".format(tokenizer.decode(r_memmap_array[8192:12288])))

                elif "label_mask" in f:
                    r_memmap_mask_array = np.memmap(filename=fpath, dtype=np.bool_, mode="r")
                    labels_mask_test1 = r_memmap_mask_array[4096:8192]
                    first_true_index1 =  np.argmax(labels_mask_test1)
                    # second_true_index1 = np.argmax(labels_mask_test1[])
                    last_true_index1 = len(labels_mask_test1) - 1 - np.argmax(np.flip(labels_mask_test1))
                    labels_mask_test2 = r_memmap_mask_array[8192:12288]
                    first_true_index2 =  np.argmax(labels_mask_test2)
                    last_true_index2 = len(labels_mask_test2) - 1 - np.argmax(np.flip(labels_mask_test2))

                    print("dir: {}\n".format(fpath))
                    print("r_memmap_mask sample0 : {}\n".format(tokenizer.decode(r_memmap_mask_array[4096:8192])))
                    # r_memmap_array = np.memmap(filename=fpath, dtype=np.uint16, mode="r")
                    print("r_memmap_mask sample1 : {}\n".format(tokenizer.decode(r_memmap_mask_array[8192:12288])))
                    
                else:
                    r_memmap_array = np.memmap(filename=fpath, dtype=np.uint16, mode="r")
                    print("dir: {}\n".format(fpath))
                    # print("r_memmap sample0 id: {}\n".format(r_memmap_array[4096:8192]))
                    
                    print("r_memmap sample0 : {}\n".format(tokenizer.decode(r_memmap_array[4096:8192], skip_special_tokens=False)))
                    # r_memmap_array = np.memmap(filename=fpath, dtype=np.uint16, mode="r")
                    print("r_memmap sample1 : {}\n".format(tokenizer.decode(r_memmap_array[8192:12288], skip_special_tokens=False)))
                
                print("="*20)
        if r_memmap_mask_array is not None:
            print("sample1 : {}\n".format(tokenizer.decode(r_memmap_array[4096:4096+last_true_index1+1], skip_special_tokens=False)))
            print("sample1 asis ans : {}\n".format(tokenizer.decode(r_memmap_array[4096+first_true_index1:4096+last_true_index1+1], skip_special_tokens=False)))
            print("sample2 : {}\n".format(tokenizer.decode(r_memmap_array[8192:8192+last_true_index2+1], skip_special_tokens=False)))
            print("sample2 asis ans : {}\n".format(tokenizer.decode(r_memmap_array[8192+first_true_index2:8192+last_true_index2+1], skip_special_tokens=False)))


    


if __name__ == "__main__":

    try:
        tknz_path, npy_path = sys.argv[1], sys.argv[2]
    except IndexError:
        raise AIGCcodeCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    print("loading tknz:")
    print(npy_path)
    print("loading npy:")
    print(tknz_path)

    print("npy loaded.")
    main(tknz_path, npy_path)
