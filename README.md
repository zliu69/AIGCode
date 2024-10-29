<div align="center">
  <br>
  <br>
  <h1>AIGCode: Language Model For Coding</h1>
</div>

AIGCode is a repository for training unbalanced MoE/PLE language models for coding task on open source data. 
Adapt from https://github.com/allenai/OLMo.

## Installation

First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system.

To install from source (recommended for training/fine-tuning) run:

```bash
git clone https://github.com/zliu69/AIGCode.git
cd AIGCode
pip install -e .[all]
```

## Models

### Overview

The core models in the AIGCode family released so far are (all trained on the [m-a-p/Matrix dataset](https://huggingface.co/datasets/m-a-p/Matrix)) 1.5T/4.7T: 
| Model | Training Tokens | Context Length | Training Config ☨ |
|-------|-----------------|:--------------:|-----------------|
| [AIGCode 7B](https://huggingface.co/zimo223/AIGCode-3B-7B-Base) | 1.5 Trillion | 4096 | [configs/official/AIGCode-7B.yaml](https://github.com/zliu69/AIGCode/blob/open_source/configs/official/AIGCode-7B.yaml) |
| [AIGCode 7B Chat](https://huggingface.co/zimo223/AIGCode-3B-7B-chat-v0.1) | 1.5 Trillion(pretrain) + 15 Billion(anneal+aft)  | 4096 | [configs/official/AIGCode-7B-sft.yaml](https://github.com/zliu69/AIGCode/blob/open_source/configs/official/AIGCode-7B-sft.yaml) |


## Inference

You can utilize Hugging Face transformers to run inference on the AIGCode Transformers after you download the model above and install aigcode & hf_aigcode:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

aigcode = AutoModelForCausalLM.from_pretrained("AIGCode-3B-7B-Base", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("AIGCode-3B-7B-Base", trust_remote_code=True)

message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
response = aigcode.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```

Alternatively, with the Hugging Face pipeline abstraction:

```python
from transformers import pipeline
aigcode_pipe = pipeline("text-generation", model="allenai/AIGCode-1.7-7B-hf")
print(aigcode_pipe("Language modeling is"))
```

## Reproducibility

### Training

The configs used to train the official AIGCode models are provided in the [`configs/official/`](https://github.com/zliu69/AIGCode/blob/open_source/configs/official) directory.

```bash
torchrun --nproc_per_node=8 scripts/train.py configs/official/AIGCode-1B.yaml
```

You can use the same method to launch multi-node jobs as well. See [the documentation](https://pytorch.org/docs/stable/elastic/run.html) for `torchrun` to understand the additional arguments you'll need to configure the rendezvous backend / endpoint.
