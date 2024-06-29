# Finetuning LLM

This repository contains the code for finetuning a language model on a custom dataset. The code is based on the [LLM-Workshop](https://github.com/pacman100/LLM-Workshop/tree/main/chat_assistant/sft) and I trained with huggingface, Unsloth with config deepspeed, fsdp. I will take notes some technique about them if I have time  . 

Firstly, you need to install the requirements:

## Installation

```shell
pip install -r requirements.txt
```

## Data
You can use dataset from huggingface on your custom dataset. You can see [prepare_dataset.py](prepare_dataset.py) for more detail what I prepare dataset with format json.

## Fine-tuning
```
CUDA_VISIBLE_DEVICES=0,1
torchrun ----nproc-per-node=2 train_unsloth_lora_1.py 
```



