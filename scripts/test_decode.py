import numpy as np
import os
import sys




npy_file_path = "/cpfs01/shared/public/zhangyangyun/workspace/python/aigcode/tokenizers/aigcode_eleuther-ai-gpt-neox-20b-pii-special.json"
#npy_file_path = "/Users/zhangyangyun/Documents/workspace/python/aigcode/tokenizers/test.json"
new_file_path = "/cpfs01/shared/public/zhangyangyun/workspace/python/aigcode/tokenizers/aigcode.json"
with open(npy_file_path, "r", encoding="UTF-8") as f:
    with open(new_file_path, "w", encoding="UTF-8") as f2:
        info = f.read()
        f2.write(info)
        f2.close()

from aigcode import Tokenizer
old_tokenizer_path = "/Users/zhangyangyun/Documents/workspace/python/aigcode/data/npy/part-0-00000.npy"
new_tokenizer_path = "/Users/zhangyangyun/Documents/workspace/python/aigcode/data/npy/new-part-0-00000.npy"

# Load the old tokenizer
old_tokenizer = Tokenizer.from_file(old_tokenizer_path)
# Load the new tokenizer
new_tokenizer = Tokenizer.from_file(new_tokenizer_path)

# 读取npy文件 mmap_mode='r' 读取文件
# 读取npy文件
mmap = np.memmap(npy_file_path, dtype=np.uint16, mode="r")
print(mmap)
# 解析
info = old_tokenizer.decode(mmap)
print(info)

