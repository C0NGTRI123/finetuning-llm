import os
import datasets
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from transformers import AutoTokenizer, TrainingArguments, Trainer
from unsloth.unsloth import FastLanguageModel
import fire


def load_model(model_name_or_path, max_seq_length):
    local_rank = os.environ["LOCAL_RANK"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=max_seq_length,
        dtype=None,
        device_map=f"cuda:{local_rank}",
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


def load_dataset(tokenizer: AutoTokenizer, dataset_name_or_path: str, max_seq_length: int):
    def formatting_prompts_func(examples):
        input_ids = [tokenizer.apply_chat_template(conversation, tokenize=True)[: max_seq_length] for conversation in
                     examples['conversations']]
        return {"input_ids": input_ids, }

    pass

    dataset = datasets.load_dataset("json", data_files=dataset_name_or_path, split="train")
    dataset = dataset.map(formatting_prompts_func, remove_columns=list(dataset.features), batched=True, )

    return dataset


def train(
    model_name_or_path: str = 'unsloth/llama-3-8b-Instruct-bnb-4bit',
    train_batch_size: int = 4,
    max_seq_length: int = 2048,
):
    dataset_name = 'llm_labelme_chat.json'
    output_dir = "unsloth-lora-checkpoints"

    # Load model and tokenizer
    tokenizer, model = load_model(model_name_or_path, max_seq_length)

    # Load dataset
    train_dataset = load_dataset(tokenizer=tokenizer, dataset_name_or_path=dataset_name, max_seq_length=max_seq_length)

    collator = DataCollatorForCompletionOnlyLM(
        instruction_template="<|im_start|>user\n",
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        fp16=False,
        bf16=True,
        learning_rate=2e-4,
        optim="adamw_8bit",
        lr_scheduler_type='linear',
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=5,
        save_total_limit=3,
        num_train_epochs=2,
        group_by_length=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
