import argparse
import json
import os
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from train import (
    load_config,
    load_and_prepare_dataset,
    RuntimeCollator,
    compute_metrics
)


def evaluate_model(cfg, checkpoint_path):

    print(f"\n{'='*60}")
    print(f"Model Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Eval subset: {cfg.eval_subset}")
    print(f"{'='*60}\n")
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    
    print("Loading and preparing evaluation dataset...")
    collator = RuntimeCollator(tokenizer, max_length=cfg.max_length)
    _, eval_set = load_and_prepare_dataset(cfg)
    
    print(f"Evaluation dataset size: {len(eval_set)}")
    
    eval_args = TrainingArguments(
        output_dir="./eval_output",
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_set,
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    
    print("\nRunning evaluation...")
    results = trainer.evaluate()
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    for metric, value in results.items():
        print(f"{metric}: {value}")
    print(f"{'='*60}\n")
    
    return results


def save_results(results, output_path="eval_results.json"):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/train.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="eval_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--eval-subset",
        type=int,
        default=None,
        help="Override number of evaluation examples"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    results = evaluate_model(cfg, args.checkpoint_path)
    save_results(results, args.output_path)
    
    print("\nEvaluation complete!")