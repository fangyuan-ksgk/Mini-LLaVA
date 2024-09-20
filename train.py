from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

class MMTrainer(Trainer):
    def __init__(self, *args, custom_train_dataloader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_train_dataloader = custom_train_dataloader

    def get_train_dataloader(self): # Wrap on customized dataloader
        return self.custom_train_dataloader


def train(model, tokenizer, train_dataloader, eval_dataloader=None, use_lora=False):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Apply LoRA if specified
    if use_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["mm_projector"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    # Create Trainer instance
    trainer = MMTrainer(
        model=model,
        args=training_args,
        custom_train_dataloader = train_dataloader,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model("./final_model")
    print("Model saved to ./final_model")