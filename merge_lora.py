from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model_path = 'unsloth/llama-3-8b-Instruct-bnb-4bit'
lora_path = 'unsloth-lora-checkpoints'
output_path = 'llama-3-8b-Instruct-bnb-4bit-chat'

base = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=torch.bfloat16,
)
lora_model = PeftModel.from_pretrained(base, lora_path)
model = lora_model.merge_and_unload()
model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_path)
