<div align="center">
  <!-- <img src="https://github.com/allenai/AIGCcode/assets/8812459/774ac485-a535-4768-8f7c-db7be20f5cc3" width="300"/> -->
  <img src="https://allenai.org/aigcode/aigcode-7b-animation.gif" alt="AIGCcode Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <br>
  <h1>AIGCcode: Open Language Model</h1>
</div>
<p align="center">
  <a href="https://github.com/allenai/AIGCcode/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/AIGCcode">
  </a>
  <a href="https://github.com/allenai/AIGCcode/releases">
    <img alt="GitHub release" src="https://img.shields.io/github/release/allenai/AIGCcode.svg">
  </a>
  <a href="https://arxiv.org/pdf/2402.00838.pdf">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2402.00838-blue">
  </a>
</p>

AIGCcode is a repository for training and using AI2's state-of-the-art open language models. 
It is built by scientists, for scientists.

## Installation

First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system.

To install from source (recommended for training/fine-tuning) run:

```bash
git clone https://github.com/allenai/AIGCcode.git
cd AIGCcode
pip install -e .[all]
```

Otherwise you can install the model code by itself directly from PyPI with:

```bash
pip install ai2-aigcode
```

## Models

### Overview

The core models in the AIGCcode family released so far are (all trained on the [Dolma dataset](https://huggingface.co/datasets/allenai/dolma)): 
| Model | Training Tokens | Context Length | Training Config | W&B Logs | Data Order File(s) ☨ |
|-------|-----------------|:--------------:|-----------------|----------|--------------------|
| [AIGCcode 1B](https://huggingface.co/allenai/AIGCcode-1B) | 3 Trillion | 2048 | [configs/official/AIGCcode-1B.yaml](https://github.com/allenai/AIGCcode/blob/main/configs/official/AIGCcode-1B.yaml) | [wandb.ai/…/AIGCcode-1B](https://wandb.ai/ai2-llm/AIGCcode-1B/reports/AIGCcode-1B--Vmlldzo2NzY1Njk1) | [epoch 1](https://aigcode-checkpoints.org/ai2-llm/aigcode-small/46zc5fly/train_data/global_indices.npy) |
| [AIGCcode 7B](https://huggingface.co/allenai/AIGCcode-7B) | 2.5 Trillion | 2048 | [configs/official/AIGCcode-7B.yaml](https://github.com/allenai/AIGCcode/blob/main/configs/official/AIGCcode-7B.yaml) | [wandb.ai/…/AIGCcode-7B](https://wandb.ai/ai2-llm/AIGCcode-7B/reports/AIGCcode-7B--Vmlldzo2NzQyMzk5) | [epoch 1](https://aigcode-checkpoints.org/ai2-llm/aigcode-medium/wvc30anm/train_data/global_indices.npy), [epoch 2](https://aigcode-checkpoints.org/ai2-llm/aigcode-medium/wd2gxrza/train_data/global_indices.npy) |
| [AIGCcode 7B Twin 2T](https://huggingface.co/allenai/AIGCcode-7B-Twin-2T) | 2 Trillion  | 2048 | [configs/official/AIGCcode-7B.yaml](https://github.com/allenai/AIGCcode/blob/main/configs/official/AIGCcode-7B.yaml) | [wandb.ai/…/AIGCcode-7B-Twin-2T](https://wandb.ai/ai2-llm/AIGCcode-7B/reports/AIGCcode-7B-Twin-2T--Vmlldzo2NzU0NTIz) | [epoch 1](https://aigcode-checkpoints.org/ai2-llm/aigcode-medium/wvc30anm/train_data/global_indices.npy) |

> ☨ *See [Inspecting training data](#inspecting-training-data) below for usage.*

### Checkpoints

URLs to checkpoints at intermediate steps of the models' trainings can be found in the csv files under [`checkpoints/official/`](https://github.com/allenai/AIGCcode/blob/main/checkpoints/official). These 'directory' URLs cannot currently be directly accessed, but files within the directory are publicly accessible. These URLs can also be provided to the training script to resume training from the checkpoint (see [Training](#training)). Each checkpoint directory consists of:

- `config.yaml`: the config at that training step.
- `model.pt`, `optim.pt`, `train.pt`: model, optimizer and training state at that training step.

Details about the other types of AIGCcode checkpoints (including AIGCcode HF Transformers checkpoints) can be found in [Checkpoints.md](https://github.com/allenai/AIGCcode/blob/main/docs/Checkpoints.md).

## Inference

You can utilize our Hugging Face integration to run inference on the AIGCcode Transformers checkpoints:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

aigcode = AutoModelForCausalLM.from_pretrained("allenai/AIGCcode-1.7-7B-hf")
tokenizer = AutoTokenizer.from_pretrained("allenai/AIGCcode-1.7-7B-hf")

message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
response = aigcode.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```

Alternatively, with the Hugging Face pipeline abstraction:

```python
from transformers import pipeline
aigcode_pipe = pipeline("text-generation", model="allenai/AIGCcode-1.7-7B-hf")
print(aigcode_pipe("Language modeling is"))
```

### Inference on finetuned checkpoints

If you finetune the model using the code in [Fine-tuning](#fine-tuning), you can use the conversion script to convert a native AIGCcode checkpoint to a Hugging Face-compatible checkpoint.

```bash
python scripts/convert_aigcode_to_hf_new.py --input_dir /path/to/aigcode/checkpoint --output_dir /path/to/hf/checkpoint/ --tokenizer_json_path tokenizers/allenai_gpt-neox-aigcode-dolma-v1_5.json
```

### Quantization

```python
aigcode = AutoModelForCausalLM.from_pretrained("allenai/AIGCcode-1.7-7B-hf", torch_dtype=torch.float16, load_in_8bit=True)  # requires bitsandbytes
```

The quantized model is more sensitive to typing / cuda, so it is recommended to pass the inputs as inputs.input_ids.to('cuda') to avoid potential issues.

## Reproducibility

### Training

The configs used to train the official AIGCcode models are provided in the [`configs/official/`](https://github.com/allenai/AIGCcode/blob/main/configs/official) directory.

Note that while the training and validation data is public and free to download, the paths to the data within those configs are pointed at a CloudFlare R2 bucket, which requires an API key for programmatic access.
So in order to use any of these configs to reproduce a training run you'll first have to download the corresponding data to a location of your choosing and then update the paths in the config accordingly.

You can derive the public HTTP URL from an R2 URL by replacing `r2://aigcode-data` with `https://aigcode-data.org`.
For example, if the R2 data URL is:

`r2://aigcode-data/preprocessed/aigcode-mix/v1_5/gpt-neox-20b-pii-special/part-000-00000.npy`

then the corresponding public URL is:

`https://aigcode-data.org/preprocessed/aigcode-mix/v1_5/gpt-neox-20b-pii-special/part-000-00000.npy`

Once you've updated the data paths in the config you can launch a training run via `torchrun`. For example, to launch the 1B model training on a single 8x GPU node, you would run:

```bash
torchrun --nproc_per_node=8 scripts/train.py configs/official/AIGCcode-1B.yaml
```

You can use the same method to launch multi-node jobs as well. See [the documentation](https://pytorch.org/docs/stable/elastic/run.html) for `torchrun` to understand the additional arguments you'll need to configure the rendezvous backend / endpoint.

To resume training from a checkpoint, you can pass its path (local or URL)
to `scripts/train.py` with the `--load_path` arguments. For example, to resume training from step 1000 of the AIGCcode 1B run:

```bash
torchrun --nproc_per_node=8 scripts/train.py configs/official/AIGCcode-1B.yaml --load_path https://aigcode-checkpoints.org/ai2-llm/aigcode-small/w1r5xfzt/step1000-unsharded
```

### Inspecting training data

You may be interested in inspecting the exact tokens that composed a particular batch during the training of one of the AIGCcode models.
We provide tools to do this, but first you'll need to download the data as above (unless you have an R2 API key) and update the corresponding config accordingly.

Then take note of the URL of the data order file you want, which can be found in the [Models Overview](#models-overview) table. For example, the data order file for the first epoch of the AIGCcode-7B model is [https://aigcode-checkpoints.org/ai2-llm/aigcode-medium/wvc30anm/train_data/global_indices.npy](https://aigcode-checkpoints.org/ai2-llm/aigcode-small/46zc5fly/train_data/global_indices.npy).

Once you have that you can use this snippet to inspect the data within a particular batch:

```python
import numpy as np
from cached_path import cached_path

from aigcode.config import TrainConfig
from aigcode.data import build_memmap_dataset

# Update these paths to what you want:
data_order_file_path = cached_path("https://aigcode-checkpoints.org/ai2-llm/aigcode-medium/wvc30anm/train_data/global_indices.npy")
train_config_path = "configs/official/AIGCcode-7B.yaml"


cfg = TrainConfig.load(train_config_path)
dataset = build_memmap_dataset(cfg, cfg.data)
batch_size = cfg.global_train_batch_size
global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)


def get_batch_instances(batch_idx: int) -> list[list[int]]:
    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    batch_indices = global_indices[batch_start:batch_end]
    batch_instances = []
    for index in batch_indices:
        token_ids = dataset[index]["input_ids"].tolist()
        batch_instances.append(token_ids)
    return batch_instances


# Get all 2048 x 2048 token IDs in the first batch.
get_batch_instances(0)
```


## Fine-tuning

To fine-tune an AIGCcode model using our trainer you'll first need to prepare your dataset by tokenizing it and saving the tokens IDs to a flat numpy memory-mapped array. See [`scripts/prepare_tulu_data.py`](./scripts/prepare_tulu_data.py) for an example with the Tulu V2 dataset, which can be easily modified for other datasets.

Next, prepare your training config. There are many examples in the [`configs/`](https://github.com/allenai/AIGCcode/blob/main/configs) directory that you can use as a starting point. The most important thing is to make sure the model parameters (the `model` field in the config) match up with the checkpoint you're starting from. To be safe you can always start from the config that comes with the model checkpoint. At a minimum you'll need to make the following changes to the config or provide the corresponding overrides from the command line:

- Update `load_path` to point to the checkpoint you want to start from.
- Set `reset_trainer_state` to `true`.
- Update `data.paths` to point to the `token_ids.npy` file you generated.
- Optionally update `data.label_mask_paths` to point to the `label_mask.npy` file you generated, unless you don't need special masking for the loss.
- Update `evaluators` to add/remove in-loop evaluations.

Once you're satisfied with your training config, you can launch the training job via `torchrun`. For example:

```
torchrun --nproc_per_node=8 scripts/train.py {path_to_train_config} \
    --data.paths=[{path_to_data}/input_ids.npy] \
    --data.label_mask_paths=[{path_to_data}/label_mask.npy] \
    --load_path={path_to_checkpoint} \
    --reset_trainer_state
```

Note: passing CLI overrides like `--reset_trainer_state` is only necessary if you didn't update those fields in your config.

## Evaluation

Additional tools for evaluating AIGCcode models are available at the [AIGCcode Eval](https://github.com/allenai/ai2-aigcode-eval) repo.

## Citing

```bibtex
@article{AIGCcode,
  title={AIGCcode: Accelerating the Science of Language Models},
  author={Dirk Groeneveld and Iz Beltagy and Pete Walsh and Akshita Bhagia and Rodney Kinney and Oyvind Tafjord and A. Jha and Hamish Ivison and Ian Magnusson and Yizhong Wang and Shane Arora and David Atkinson and Russell Authur and Khyathi Raghavi Chandu and Arman Cohan and Jennifer Dumas and Yanai Elazar and Yuling Gu and Jack Hessel and Tushar Khot and William Merrill and Jacob Daniel Morrison and Niklas Muennighoff and Aakanksha Naik and Crystal Nam and Matthew E. Peters and Valentina Pyatkin and Abhilasha Ravichander and Dustin Schwenk and Saurabh Shah and Will Smith and Emma Strubell and Nishant Subramani and Mitchell Wortsman and Pradeep Dasigi and Nathan Lambert and Kyle Richardson and Luke Zettlemoyer and Jesse Dodge and Kyle Lo and Luca Soldaini and Noah A. Smith and Hanna Hajishirzi},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:267365485},
  journal={arXiv preprint},
}
```
