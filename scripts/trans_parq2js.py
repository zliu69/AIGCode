import datasets
from datasets import load_dataset
import multiprocessing as mp
import json
import numpy as np
import os
import click
from typing import Generator, List, Optional, Sequence, Tuple, TypeVar, Union
from smashed.utils.io_utils import recursively_list_files




@click.command()
@click.argument(
    "src",
    nargs=-1,
    type=str,
    required=True,
)
@click.option(
    "-o",
    "--output",
    type=str,
    help="Specify the output path.",
    prompt="Output directory",
)
@click.option("-j", "--workers", "max_workers", type=int, default=1, help="Defaults to number of CPUs")

def main(
    src: Tuple[str, ...],
    output: str,
    max_workers: int = 1,
):
    print("WARNING: THIS SCRIPT IS DEPRECATED!!!")
    print(
        "Consider using the tokenization tool in the Dolma toolkit: "
        "https://github.com/allenai/dolma/blob/main/docs/tokenize.md"
    )

    # if not ack_deprecated:
    #     continue_question = input("Do you want to continue? [y/N]: ")
    #     if not (c := continue_question.lower().strip()) or c != "y":
    #         print("Aborting.")
    #         return

    print("=== CONFIGURATION ===")
    print(f"src:              {src}")
    print(f"output:           {output}")
    print(f"max_workers:      {max_workers}")
    print("=====================")


    exploded_src = list(set(path for prefix in src for path in recursively_list_files(prefix) if path.endswith(".parquet")))
    print("exploded_src: {}\n".format(exploded_src))
    # output_path = output.as_path
            # make sure the directory exists
    # self._local_path.parent.mkdir(parents=True, exist_ok=True)
    os.makedirs(output, exist_ok=True)

    for parq_f in exploded_src:
        output_file = os.path.join(output, "{}.json".format(parq_f.rsplit(".",1)[0].rsplit("/",1)[1]))
        print("output_file: {}\n".format(output_file))
        print("parquet_file: {}\n".format(parq_f))

        # path.endswith(".parquet"):
        # 读取 Parquet 文件

        ds_train = load_dataset('parquet', data_files=parq_f)["train"]
        # 遍历 DataFrame 的每一行
        js_lst = []
        try:
            if "question" in ds_train.column_names and "answer" in ds_train.column_names:
                for js in ds_train:
                    js_lst.append(js)
            
                print("js len: {}\n".format(len(js_lst)))
                with open(output_file, "a+", encoding='utf-8') as f:
                    f.write("\n".join([json.dumps(j, ensure_ascii=False) for j in js_lst]))

        except Exception as e:
                # log.error(f"Error processing {parq_f} -> {output_file}")
                print(f"Error processing {parq_f} -> {output_file}")
                print(e)
                pass
        
    # exploded_src, exploded_dst = make_source_and_target(
    #     src=src, output=output, random_seed=random_seed, paths_per_worker=paths_per_worker
    # )

    # creating a partial here with all the arguments we need to pass to fill_memmap except for the paths
    # so that we don't make mistakes between debug and non-debug mode
    # fill_memmap_fn = functools.partial(
    #     fill_memmap,
    #     tokenizer_id=tokenizer_id,
    #     dtype=dtype,
    #     max_tokens=max_tokens,
    #     safe_mode=safe_mode,
    #     sample_rate=sample_rate,
    #     random_seed=random_seed,
    #     repeat_sequence=repeat_sequence,
    #     cache_dir=cache_dir,
    # )

    


if __name__ == "__main__":
    mp.set_start_method("spawn")

    main()