"""
Model Evaluation Script

This script provides comprehensive evaluation of machine unlearning models
on multiple datasets using various metrics.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import pandas as pd

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from config import TrainConfig, EvaluationConfig
from model_utils import load_model_and_tokenizer
from evaluation.metrics import (
    evaluate_on_truthfulqa,
    evaluate_on_wikipedia,
    compare_models,
    evaluate_model_comprehensive
)


def load_model_from_checkpoint(checkpoint_path: str, config: TrainConfig):
    """Load model from checkpoint."""
    model, tokenizer = load_model_and_tokenizer(config.model_name, config.cache_dir)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print("Using base model (no checkpoint loaded)")
    
    return model, tokenizer


def evaluate_single_model(
    model_path: str,
    config: TrainConfig,
    eval_config: EvaluationConfig,
    datasets: dict,
    output_dir: str,
    model_name: str = "model"
):
    """Evaluate a single model on specified datasets."""
    
    # Load model
    model, tokenizer = load_model_from_checkpoint(model_path, config)
    
    results = {}
    
    for dataset_name, dataset_info in datasets.items():
        print(f"\\nEvaluating on {dataset_name}...")
        
        if dataset_name == "truthfulqa":
            eval_results = evaluate_on_truthfulqa(
                model=model,
                tokenizer=tokenizer,
                config=eval_config,
                max_samples=dataset_info.get("max_samples")
            )
        else:
            eval_results = evaluate_on_wikipedia(
                model=model,
                tokenizer=tokenizer,
                dataset_path=dataset_info["path"],
                config=eval_config,
                max_samples=dataset_info.get("max_samples")
            )
        
        results[dataset_name] = eval_results
        
        # Save detailed results
        if "results_df" in eval_results:
            output_path = f"{output_dir}/{model_name}_{dataset_name}_detailed.csv"
            eval_results["results_df"].to_csv(output_path, index=False)
            print(f"Detailed results saved: {output_path}")
    
    # Save aggregate metrics
    aggregate_results = []
    for dataset_name, eval_results in results.items():
        if "aggregate_metrics" in eval_results:
            metrics = eval_results["aggregate_metrics"].copy()
            metrics["dataset"] = dataset_name
            metrics["model"] = model_name
            aggregate_results.append(metrics)
    
    if aggregate_results:
        aggregate_df = pd.DataFrame(aggregate_results)
        aggregate_path = f"{output_dir}/{model_name}_aggregate_metrics.csv"
        aggregate_df.to_csv(aggregate_path, index=False)
        print(f"\\nAggregate metrics saved: {aggregate_path}")
        
        # Print summary
        print(f"\\n=== {model_name} Results Summary ===")
        for _, row in aggregate_df.iterrows():
            print(f"{row['dataset']}: BLEU={row['BLEU']:.4f}, ROUGE-L={row['ROUGE-L']:.4f}, BERTScore-F1={row['BERTScore-F1']:.4f}")
    
    return results


def compare_base_vs_unlearned(
    base_model_path: str,
    unlearned_model_path: str,
    config: TrainConfig,
    eval_config: EvaluationConfig,
    datasets: dict,
    output_dir: str
):
    """Compare base model against unlearned model."""
    
    print("Loading models for comparison...")
    
    # Load base model
    base_model, tokenizer = load_model_from_checkpoint(base_model_path, config)
    
    # Load unlearned model  
    unlearned_model, _ = load_model_from_checkpoint(unlearned_model_path, config)
    
    # Prepare datasets for comparison function
    dataset_objects = {}
    for name, info in datasets.items():
        if name == "truthfulqa":
            try:
                from datasets import load_dataset
                dataset_objects[name] = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
            except:
                print(f"Could not load {name}, skipping...")
                continue
        else:
            dataset_objects[name] = info["path"]
    
    # Run comparison
    comparison_results = compare_models(
        base_model=base_model,
        unlearned_model=unlearned_model,
        tokenizer=tokenizer,
        datasets=dataset_objects,
        config=eval_config
    )
    
    # Process and save results
    for model_name, model_results in comparison_results.items():
        for dataset_name, eval_results in model_results.items():
            if "results_df" in eval_results:
                output_path = f"{output_dir}/comparison_{model_name}_{dataset_name}.csv"
                eval_results["results_df"].to_csv(output_path, index=False)
    
    # Create comparison summary
    summary_data = []
    for model_name, model_results in comparison_results.items():
        for dataset_name, eval_results in model_results.items():
            if "aggregate_metrics" in eval_results:
                metrics = eval_results["aggregate_metrics"].copy()
                metrics["model"] = model_name
                metrics["dataset"] = dataset_name
                summary_data.append(metrics)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = f"{output_dir}/comparison_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print("\\n=== Comparison Results ===")
        for dataset in summary_df["dataset"].unique():
            dataset_data = summary_df[summary_df["dataset"] == dataset]
            print(f"\\n{dataset.upper()}:")
            for _, row in dataset_data.iterrows():
                print(f"  {row['model']}: BLEU={row['BLEU']:.4f}, ROUGE-L={row['ROUGE-L']:.4f}, BERTScore-F1={row['BERTScore-F1']:.4f}")
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate machine unlearning models")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--base_model_path", type=str, help="Path to base model checkpoint (for comparison)")
    parser.add_argument("--unlearned_model_path", type=str, help="Path to unlearned model checkpoint (for comparison)")
    parser.add_argument("--output_dir", type=str, default="./results/evaluation", help="Output directory for results")
    parser.add_argument("--config_path", type=str, help="Path to config file (optional)")
    parser.add_argument("--max_samples", type=int, help="Maximum samples per dataset")
    parser.add_argument("--datasets", nargs="+", default=["truthfulqa", "wikipedia_person_unlearn"], help="Datasets to evaluate on")
    parser.add_argument("--compare", action="store_true", help="Compare base vs unlearned models")
    
    args = parser.parse_args()
    
    # Load configuration
    config = TrainConfig()
    eval_config = EvaluationConfig()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare datasets
    datasets = {}
    if "truthfulqa" in args.datasets:
        datasets["truthfulqa"] = {"max_samples": args.max_samples}
    
    if "wikipedia_person_unlearn" in args.datasets:
        datasets["wikipedia_person_unlearn"] = {
            "path": config.wiki_dataset_path,
            "max_samples": args.max_samples
        }
    
    if "validation" in args.datasets:
        datasets["validation"] = {
            "path": config.val_dataset_path,
            "max_samples": args.max_samples
        }
    
    # Run evaluation
    if args.compare and args.base_model_path and args.unlearned_model_path:
        compare_base_vs_unlearned(
            base_model_path=args.base_model_path,
            unlearned_model_path=args.unlearned_model_path,
            config=config,
            eval_config=eval_config,
            datasets=datasets,
            output_dir=args.output_dir
        )
    elif args.model_path:
        evaluate_single_model(
            model_path=args.model_path,
            config=config,
            eval_config=eval_config,
            datasets=datasets,
            output_dir=args.output_dir,
            model_name=os.path.basename(args.model_path).replace(".pt", "")
        )
    else:
        # Evaluate base model
        evaluate_single_model(
            model_path=None,  # Uses base model
            config=config,
            eval_config=eval_config,
            datasets=datasets,
            output_dir=args.output_dir,
            model_name="base_model"
        )
    
    print(f"\\nEvaluation complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
