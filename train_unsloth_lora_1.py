import os
import datasets
from trl import SFTTrainer
from transformers import AutoTokenizer, TrainingArguments
from unsloth.unsloth import FastLanguageModel
from unsloth.unsloth import get_chat_template
from accelerate import PartialState
import fire


def load_model(model_name_or_path, max_seq_length):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=max_seq_length,
        dtype=None,
        device_map={"": PartialState().process_index},
        load_in_4bit=True,
        use_cache=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    return tokenizer, model


def load_dataset(tokenizer: AutoTokenizer, dataset_name_or_path: str):
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts, }

    pass
    dataset = datasets.load_dataset(dataset_name_or_path, split="train")
    # dataset = datasets.load_dataset("json", data_files=dataset_name_or_path, split="train")
    dataset = dataset.map(formatting_prompts_func, remove_columns=list(dataset.features), batched=True, )

    return dataset


def train(
    model_name_or_path: str = 'unsloth/llama-3-8b-Instruct-bnb-4bit',
    train_batch_size: int = 4,
    max_seq_length: int = 2048,
):
    dataset_name = 'philschmid/guanaco-sharegpt-style'
    output_dir = "unsloth-lora-checkpoints"

    # Load model and tokenizer
    tokenizer, model = load_model(model_name_or_path, max_seq_length)

    # Load dataset
    train_dataset = load_dataset(tokenizer=tokenizer, dataset_name_or_path=dataset_name)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        learning_rate=2e-4,
        optim="adamw_8bit",
        lr_scheduler_type='linear',
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=5,
        save_total_limit=3,
        num_train_epochs=2,
        group_by_length=True,
        fp16=True,
        bf16=False,
        report_to=[]
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        packing=False,
        dataset_num_proc=2,
        max_seq_length=max_seq_length,
        dataset_text_field="text"
    )
    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
