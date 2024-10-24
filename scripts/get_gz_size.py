import os

def get_directory_size(directory):
    total_size = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return total_size

# 指定目录路径
directory_path = '/sharedata/zimoliu/data/neo_test_data_sc/book/book_all'

# 计算目录整体大小
directory_size = get_directory_size(directory_path)
# directory_size = get_directory_size(directory_path)

print(f"The total size of the directory is: {directory_size} bytes")