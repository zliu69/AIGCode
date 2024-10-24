import gzip
import time
# from itertools import (takewhile, repeat)

def count_lines_gz(file_path):
    """
    计算.gz文件的行数。
    
    :param file_path: .gz文件的路径。
    :return: 文件的行数。
    """
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        return sum(1 for _ in f)

# def iter_count(file_name):

#     buffer = 1024 * 1024 * 1024
#     # with open(file_name) as f:
#     with gzip.open(file_path, 'rt', encoding='utf-8') as f:

#         buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
#         return sum(buf.count('\n') for buf in buf_gen)


# 记录脚本开始时间
start_time = time.time()
# 使用示例
file_path = '/sharedata/zimoliu/data/neo_test_data_sc/book/book_all.0010.jsonl.gz'  # 指定.gz文件的路径
line_count = count_lines_gz(file_path)
# line_count = iter_count(file_path)
print(f"The file {file_path} has {line_count:,} lines.")

end_time = time.time()

# 计算总运行时间
total_run_time = end_time - start_time

print(f"总运行时间：{total_run_time} 秒")