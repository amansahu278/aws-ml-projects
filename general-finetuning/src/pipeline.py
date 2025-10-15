import sagemaker
from sagemaker.huggingface import HuggingFace
import argparse
import os

session = sagemaker.Session()
region = session.boto_session.region_name
role = sagemaker.get_execution_role()

# Get S3 bucket from environment variable or use SageMaker default
bucket_name = os.environ.get("SAGEMAKER_BUCKET", session.default_bucket())
project_prefix =  "general-finetuning"

model_output_path = f"s3://{bucket_name}/{project_prefix}/outputs"
code_location = f"s3://{bucket_name}/{project_prefix}/code"


def create_training_job(
    config_path: str = "configs/train.yaml",
    instance_type: str = "ml.t2.medium",
    instance_count: int = 1,
    job_name: str = None,
    model_name: str = None,
    learning_rate: float = None,
    max_steps: int = None,
    per_device_train_batch_size: int = None,
    per_device_eval_batch_size: int = None,
    warmup_steps: int = None,
    max_length: int = None,
    train_subset: int = None,
    eval_subset: int = None,
    prompt_template: str = None,
    fp16: bool = None,
):
    
    hyperparameters = {"config-path": config_path}
    
    overrides = []
    if model_name is not None:
        hyperparameters["model-name"] = model_name
        overrides.append(f"model-name={model_name}")
    if learning_rate is not None:
        hyperparameters["learning-rate"] = learning_rate
        overrides.append(f"learning-rate={learning_rate}")
    if max_steps is not None:
        hyperparameters["max-steps"] = max_steps
        overrides.append(f"max-steps={max_steps}")
    if per_device_train_batch_size is not None:
        hyperparameters["per-device-train-batch-size"] = per_device_train_batch_size
        overrides.append(f"per-device-train-batch-size={per_device_train_batch_size}")
    if per_device_eval_batch_size is not None:
        hyperparameters["per-device-eval-batch-size"] = per_device_eval_batch_size
        overrides.append(f"per-device-eval-batch-size={per_device_eval_batch_size}")
    if warmup_steps is not None:
        hyperparameters["warmup-steps"] = warmup_steps
        overrides.append(f"warmup-steps={warmup_steps}")
    if max_length is not None:
        hyperparameters["max-length"] = max_length
        overrides.append(f"max-length={max_length}")
    if train_subset is not None:
        hyperparameters["train-subset"] = train_subset
        overrides.append(f"train-subset={train_subset}")
    if eval_subset is not None:
        hyperparameters["eval-subset"] = eval_subset
        overrides.append(f"eval-subset={eval_subset}")
    if prompt_template is not None:
        hyperparameters["prompt-template"] = prompt_template
        overrides.append(f"prompt-template={prompt_template}")
    if fp16 is not None:
        hyperparameters["fp16"] = "" if fp16 else None
        overrides.append(f"fp16={fp16}")
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Config file: {config_path}")
    if overrides:
        print(f"\nHyperparameter overrides:")
        for override in overrides:
            print(f"  - {override}")
    else:
        print("No hyperparameter overrides (using config defaults)")
    print(f"{'='*60}\n")
    
    estimator = HuggingFace(
        entry_point="train.py",
        source_dir="src",
        instance_type=instance_type,
        instance_count=instance_count,
        role=role,
        transformers_version="4.36.0",
        pytorch_version="2.1.0",
        py_version="py310",
        hyperparameters=hyperparameters,
        output_path=model_output_path,
        code_location=code_location,
        base_job_name=job_name or "general-finetuning",
        sagemaker_session=session,
    )
    
    print(f"Starting training job...")
    print(f"Instance type: {instance_type}")
    print(f"Output path: {model_output_path}\n")
    
    estimator.fit(wait=True)
    
    print(f"\nTraining completed!")
    print(f"Model artifacts: {estimator.model_data}")
    
    return estimator


def create_evaluation_job(
    checkpoint_path: str,
    config_path: str = "configs/train.yaml",
    instance_type: str = "ml.m5.xlarge",
    job_name: str = None,
    eval_subset: int = None,
):
    
    hyperparameters = {
        "config-path": config_path,
        "checkpoint-path": checkpoint_path,
        "output-path": "eval_results.json",
    }
    
    if eval_subset is not None:
        hyperparameters["eval-subset"] = eval_subset
    
    print(f"\n{'='*60}")
    print(f"Evaluation Configuration")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config file: {config_path}")
    print(f"{'='*60}\n")
    
    estimator = HuggingFace(
        entry_point="evaluation.py",
        source_dir="src",
        instance_type=instance_type,
        instance_count=1,
        role=role,
        transformers_version="4.36.0",
        pytorch_version="2.1.0",
        py_version="py310",
        hyperparameters=hyperparameters,
        output_path=model_output_path,
        code_location=code_location,
        base_job_name=job_name or "model-evaluation",
        sagemaker_session=session,
    )
    
    print(f"Starting evaluation job...")
    print(f"Instance type: {instance_type}\n")
    
    estimator.fit(wait=True)
    
    print(f"\nEvaluation completed!")
    print(f"Results: {estimator.model_data}")
    
    return estimator


def run_pipeline(args):
    print("=" * 80)
    print("Starting Fine-Tuning Pipeline")
    print("=" * 80)
    print("\nPipeline steps: Training → Evaluation")
    print("=" * 80 + "\n")
    
    print("=" * 80)
    print("STEP 1: Training Model")
    print("=" * 80)
    
    train_estimator = create_training_job(
        config_path=args.config_path,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        job_name=args.job_name,
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        train_subset=args.train_subset,
        eval_subset=args.eval_subset,
        prompt_template=args.prompt_template,
        fp16=args.fp16,
    )
    
    print("\n" + "=" * 80)
    print("STEP 2: Evaluating Model")
    print("=" * 80)
    
    eval_estimator = create_evaluation_job(
        checkpoint_path=train_estimator.model_data,
        config_path=args.config_path,
        instance_type=args.eval_instance_type,
        job_name=f"{args.job_name}-eval" if args.job_name else None,
        eval_subset=args.eval_subset_override,
    )
    
    print("\n" + "=" * 80)
    print("Pipeline Completed Successfully!")
    print("=" * 80)
    print(f"\nTraining output:   {train_estimator.model_data}")
    print(f"Evaluation output: {eval_estimator.model_data}")
    print("\n" + "=" * 80)
    
    return train_estimator, eval_estimator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fine-tuning pipeline on SageMaker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    infra = parser.add_argument_group('Infrastructure')
    infra.add_argument("--config-path", type=str, default="configs/train.yaml",
                       help="Path to training config file")
    infra.add_argument("--instance-type", type=str, default="ml.t2.medium",
                       help="Instance type for training")
    infra.add_argument("--instance-count", type=int, default=1,
                       help="Number of training instances")
    infra.add_argument("--eval-instance-type", type=str, default="ml.t2.medium",
                       help="Instance type for evaluation (CPU is fine)")
    infra.add_argument("--job-name", type=str, default=None,
                       help="Custom job name prefix")
    
    hp = parser.add_argument_group('Hyperparameters (override config file)')
    hp.add_argument("--model-name", type=str, default=None,
                    help="Model to fine-tune")
    hp.add_argument("--learning-rate", type=float, default=None,
                    help="Learning rate")
    hp.add_argument("--max-steps", type=int, default=None,
                    help="Maximum training steps")
    hp.add_argument("--per-device-train-batch-size", type=int, default=None,
                    help="Training batch size per device")
    hp.add_argument("--per-device-eval-batch-size", type=int, default=None,
                    help="Evaluation batch size per device")
    hp.add_argument("--warmup-steps", type=int, default=None,
                    help="Number of warmup steps")
    hp.add_argument("--max-length", type=int, default=None,
                    help="Maximum sequence length")
    hp.add_argument("--train-subset", type=int, default=None,
                    help="Number of training examples")
    hp.add_argument("--eval-subset", type=int, default=None,
                    help="Number of eval examples")
    hp.add_argument("--prompt-template", type=str, default=None,
                    choices=["alpaca", "vicuna"],
                    help="Prompt template")
    hp.add_argument("--fp16", action="store_true", default=None,
                    help="Use FP16 training")
    
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument("--eval-subset-override", type=int, default=None,
                            help="Override eval subset size for evaluation step")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_estimator, eval_estimator = run_pipeline(args)
