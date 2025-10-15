
import yaml
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
import torch
import os
from prompt_template import PromptTemplate
import argparse
import json
from pathlib import Path

@dataclass
class TrainConfig:
    model_name: str
    output_dir: str
    train_subset: int
    eval_subset: int
    max_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    prompt_template: str
    learning_rate: float
    warmup_steps: int
    logging_steps: int
    save_steps: int
    save_total_limit: int
    eval_steps: int
    seed: int
    fp16: bool
    max_length: int

def load_config(args=None) -> TrainConfig:
    args = parse_args() if args is None else args

    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)

    valid_fields = set(TrainConfig.__annotations__.keys())

    # Making config file more flexible, so we can override with arguments
    # We only override fields that exist in TrainConfig
    for key, value in vars(args).items():
        if value is not None and key in valid_fields:
            cfg[key] = value

    return TrainConfig(**cfg)


def load_and_prepare_dataset(cfg: TrainConfig) -> tuple[Dataset, Dataset]:

    raw_dataset = load_dataset("yahma/alpaca-cleaned")

    train_set = raw_dataset["train"].select(range(cfg.train_subset))
    eval_set = raw_dataset["train"].select(
        range(cfg.train_subset, cfg.train_subset + cfg.eval_subset)
    )

    template = PromptTemplate.from_name(cfg.prompt_template)

    def format(example):
        instruction = example.get("instruction", "")
        input = example.get("input", "")
        output = example.get("output", "")
        
        text = template.format(instruction, input, output)
        return {"text": text}
    
    train_set = train_set.map(format)
    eval_set = eval_set.map(format)

    return train_set, eval_set


class RuntimeCollator:

    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch):
        texts = [item["text"] for item in batch]
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        #NOTE: 2SELF: Here, we could add mask so as to only predict the response
        encodings["labels"] = encodings["input_ids"].clone()
        encodings["labels"][encodings["labels"] == self.tokenizer.pad_token_id] = -100 # Pytorch convention.
        return encodings


def compute_metrics(eval_preds):

    logits, labels = eval_preds

    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1]) #[1, 2, 3, 4]
    shift_labels = labels[..., 1:].reshape(-1) # [2, 3, 4, 5]

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100) #label with value -100 is the pad token, ignore it in the loss.
    loss = loss_fn(torch.tensor(shift_logits), torch.tensor(shift_labels)).item()
    perplexity = torch.exp(torch.tensor(loss))
    print(json.dumps({"perplexity": perplexity}))
    return {"perplexity": perplexity}


def train_model(cfg, train_set, eval_set):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    
    collator = RuntimeCollator(tokenizer, max_length=cfg.max_length)

    train_args = TrainingArguments(
        output_dir=cfg.output_dir,
        eval_strategy="steps", # We're not using epochs, just steps
        save_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=2,
        max_steps=cfg.max_steps, # max_steps is used instead of num_train_epochs, do max_steps amt of updates and then stop training
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        fp16=cfg.fp16,
        report_to="none",
        load_best_model_at_end=True,
        seed=cfg.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # trainer.save_model(cfg.output_dir)

    model_dir = os.environ.get("SM_MODEL_DIR", cfg.output_dir)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)


def parse_args():
    
    parser = argparse.ArgumentParser(description="Train LLM with SageMaker")
    parser.add_argument("--config-path", type=str, default="configs/train.yaml")
    parser.add_argument("--model-name", type=str, default="distilgpt2")
    parser.add_argument("--train-subset", type=int, default=100)
    parser.add_argument("--eval-subset", type=int, default=20)
    parser.add_argument("--prompt-template", type=str, default="alpaca")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--fp16", action="store_true")
    
    return parser.parse_args()

if __name__ == "__main__":

    cfg = load_config(parse_args())

    train_set, eval_set = load_and_prepare_dataset(cfg)
    train_model(cfg, train_set, eval_set)