"""
Split gzipped files into smaller gzipped files.

Author: @soldni
"""

import concurrent.futures
import gzip
import os
from contextlib import ExitStack

import click

from rich.progress import (
    Progress,
    BarColumn,
    # TaskProgressColumn,
    TimeElapsedColumn,
)

from smashed.utils.io_utils import (
    # MultiPath,
    # decompress_stream,
    # open_file_for_write,
    recursively_list_files,
    # stream_file_for_read,
)

MAX_SIZE_4_GB = 4 * 1024 * 1024 * 1024
MAX_SIZE_1_GB = 1 * 1024 * 1024 * 1024
# MAX_SIZE_025_GB = 256 * 1024 * 1024

def get_directory_size(directory):
    total_size = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


def split_gzip_file(input_file, output_base, base_name, size_limit, output_ext):
    print(f"Splitting {input_file} into {output_base} with size limit {size_limit:,}")
    total_b_size = os.path.getsize(input_file)
    os.makedirs(output_base, exist_ok=True)
    # written_b_size = 0
    with ExitStack() as stack, gzip.open(input_file, "rt") as f:
        count = 0
        path = f"{output_base}/{base_name}_{count:04d}{output_ext}"
        output = stack.enter_context(gzip.open(path, "wt"))
        current_size = 0
        with Progress(
            "[progress.description]{task.description}",
            "[progress.percentage]{task.percentage:>3.2f}%",
            BarColumn(bar_width=None),
            # BarColumn(),
            # TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Splitting {}: ".format(input_file), total=float(total_b_size))
            for line in f:
                line_size = len(line)
                if current_size + line_size > size_limit:
                    stack.pop_all().close()
                    count += 1
                    print(f"Wrote {path}")
                    path = f"{output_base}/{base_name}_{count:04d}{output_ext}"
                    output = stack.enter_context(gzip.open(path, "wt"))
                    current_size = 0
                    
                output.write(str(line))
                current_size += line_size

                progress.update(
                    task, 
                    # advance=float(line_size)
                    completed=float(get_directory_size(output_base))
                    )  # 更新进度条
                # written_b_size += line_size  # 更新已写入的字节大小
            # progress.stop_task(task)
            print(f"Wrote {path}")
            stack.pop_all().close()
            print(f"Finished splitting {input_file} into {count + 1:,} files")


def process_file(file_name, input_ext, input_dir, output_dir, output_ext, size_limit):
    input_file = os.path.join(input_dir, file_name)
    if file_name.endswith(input_ext) and os.path.isfile(input_file):
        base_name = file_name.rstrip(input_ext).split("/")[-1]
        print(base_name)
        output_base = os.path.join(output_dir, base_name.split(".")[0] + "_" + base_name.split(".")[1])
        # os.path.join(output_dir, base_name)
        split_gzip_file(input_file, output_base, base_name, size_limit, output_ext)


@click.command()
@click.option("--input_dir", required=True, help="Path to input directory containing gzip files")
@click.option("--input_ext", default=".gz", help="Extension of the input files")
@click.option("--output_dir", required=True, help="Path to output directory for the split files")
@click.option("--output_ext", default=".gz", help="Extension of the output files")
@click.option("--size_limit", default=MAX_SIZE_1_GB, help="Size limit for each split file in bytes")
@click.option("--max_workers", default=1, help="Defaults to number of CPUs")
def main(input_dir: str, input_ext: str, output_dir: str, output_ext: str, size_limit: int, max_workers: int):
    os.makedirs(output_dir, exist_ok=True)
    # files_to_process = [
    #     file_name
    #     for file_name in os.listdir(input_dir)
    #     if file_name.endswith(input_ext) and os.path.isfile(os.path.join(input_dir, file_name))
    # ]

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     executor.map(process_file, files_to_process)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用executor.map并行处理任务
        # executor.map(process_file, files_to_process)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # list(executor.map(process_file, files_to_process))
        # 准备参数列表，每个元素是一个包含所有所需参数的元组
        tasks = [
            (file_name, input_ext, input_dir, output_dir, output_ext, size_limit)
            # for file_name in os.listdir(input_dir)
            for file_name in (path for path in recursively_list_files(input_dir))
            if file_name.endswith(input_ext)
        ]

        # 使用 executor.submit 提交任务，创建 Future 对象
        futures = [executor.submit(process_file, *task_args) for task_args in tasks]

        # 等待所有任务完成并获取结果
        # for future in concurrent.futures.as_completed(futures):
        #     result = future.result()
            # 可以在这里处理结果


if __name__ == "__main__":
    main()
