import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = 'llama-3-8b-Instruct-bnb-4bit-chat'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    use_cache=True,
)

messages = [
    {"role": "user", "content": "What is a famous tall tower in Paris?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Must add for generation
    return_tensors="pt",
).to("cuda")

out_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=256,
    do_sample=True,
    top_p=0.9,
    top_k=40,
    temperature=0.1,
    repetition_penalty=1.05,
)

assistant = tokenizer.batch_decode(out_ids[:, input_ids.size(1):], skip_special_tokens=True)[0].strip()
print("Assistant: ", assistant)
