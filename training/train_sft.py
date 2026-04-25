#!/usr/bin/env python3
"""SFT training script using Unsloth for efficient fine-tuning."""

import argparse
import json
from pathlib import Path


def load_dataset(path: str) -> list[dict]:
    """Load JSONL dataset."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_for_training(examples: list[dict]) -> list[dict]:
    """Format examples for SFT training."""
    formatted = []
    for ex in examples:
        # The dataset is already in chat format with messages
        formatted.append({"messages": ex["messages"]})
    return formatted


def main():
    parser = argparse.ArgumentParser(description="SFT training for Sysadmin agent")
    parser.add_argument("--train", type=str, default="dataset/sft_train.jsonl",
                        help="Training data path")
    parser.add_argument("--val", type=str, default="dataset/sft_val.jsonl",
                        help="Validation data path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct",
                        help="Base model name")
    parser.add_argument("--output", type=str, default="checkpoints/sft",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--wandb-project", type=str, default="sysadmin-game",
                        help="W&B project name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    args = parser.parse_args()

    # Check if dataset exists
    if not Path(args.train).exists():
        print(f"Error: Training data not found at {args.train}")
        print("Please create the SFT dataset first.")
        return

    print(f"Loading training data from {args.train}")
    train_data = load_dataset(args.train)
    print(f"Loaded {len(train_data)} training examples")

    val_data = []
    if Path(args.val).exists():
        val_data = load_dataset(args.val)
        print(f"Loaded {len(val_data)} validation examples")

    # Import training libraries
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
        import torch
    except ImportError as e:
        print(f"Error importing training libraries: {e}")
        print("Install with: uv sync --extra train")
        return

    # Initialize W&B
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name="sft-training")
        except Exception as e:
            print(f"W&B init failed: {e}, continuing without logging")

    # Load model with Unsloth
    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Format data
    train_formatted = format_for_training(train_data)
    val_formatted = format_for_training(val_data) if val_data else None

    # Create datasets
    def formatting_func(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    train_dataset = Dataset.from_list(train_formatted)
    train_dataset = train_dataset.map(formatting_func, batched=True)

    eval_dataset = None
    if val_formatted:
        eval_dataset = Dataset.from_list(val_formatted)
        eval_dataset = eval_dataset.map(formatting_func, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=50,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=50 if eval_dataset else None,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=True,
        report_to="wandb" if not args.no_wandb else "none",
        save_total_limit=2,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    print(f"Saving model to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("Training complete!")


if __name__ == "__main__":
    main()
