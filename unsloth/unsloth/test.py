from unsloth.unsloth.chat_templates import get_chat_template
from transformers import AutoTokenizer
import datasets
from unsloth.unsloth import FastLanguageModel


# def load_dataset(tokenizer: AutoTokenizer, dataset_name_or_path: str):
#     tokenizer = get_chat_template(
#         tokenizer,
#         chat_template="llama-3",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
#         mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style
#     )
#
#     def formatting_prompts_func(examples):
#         convos = examples["conversations"]
#         texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
#         return {"text": texts, }
#
#     pass
#
#     dataset = datasets.load_dataset("json", data_files=dataset_name_or_path, split="train")
#     dataset = dataset.map(formatting_prompts_func, batched=True, )
#
#     return dataset

def load_dataset(tokenizer: AutoTokenizer, dataset_name_or_path: str):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts, }

    pass

    dataset = datasets.load_dataset("json", data_files=dataset_name_or_path, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True, )

    return dataset


model_name_or_path = 'unsloth/llama-3-8b-Instruct-bnb-4bit'
dataset_name = 'llm_labelme.json'
_, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=2048,
        dtype=None,
    )

print(tokenizer)

