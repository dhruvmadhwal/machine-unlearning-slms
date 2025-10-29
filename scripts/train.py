"""
Enhanced Training Script for Machine Unlearning

This script supports both random labelling and gradient ascent unlearning methods
with configurable parameters and proper checkpoint management.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import TrainConfig
from model_utils import load_model_and_tokenizer
from unlearning.random_labelling import RandomLabellingUnlearner
from unlearning.gradient_ascent import GradientAscentUnlearner


def main():
    parser = argparse.ArgumentParser(description="Train machine unlearning models")
    parser.add_argument("--method", type=str, default="random_labelling", 
                       choices=["random_labelling", "gradient_ascent"],
                       help="Unlearning method to use")
    parser.add_argument("--data_path", type=str, help="Path to training data")
    parser.add_argument("--model_name", type=str, 
                       choices=["nvidia/Nemotron-Mini-4B-Instruct", 
                               "meta-llama/Llama-3.2-3B-Instruct", 
                               "microsoft/Phi-3.5-mini-instruct"],
                       help="Model name to use")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = TrainConfig()
    
    # Override config with command line arguments
    if args.method:
        config.unlearning_method = args.method
    if args.data_path:
        config.train_dataset_path = args.data_path
    if args.model_name:
        config.set_model(args.model_name)
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size_training = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.output_dir:
        config.checkpoints_dir = args.output_dir

    print(f"Starting {config.unlearning_method} training...")
    print(f"Model: {config.model_name}")
    print(f"Data: {config.train_dataset_path}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size_training}")
    print(f"Learning rate: {config.lr}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.model_name, config.cache_dir)
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    
    # Select and run unlearning method
    if config.unlearning_method == "random_labelling":
        unlearner = RandomLabellingUnlearner(config)
        epoch_losses = unlearner.unlearn(
            model=model,
            tokenizer=tokenizer,
            random_data_path=config.train_dataset_path,
            optimizer=optimizer
        )
    elif config.unlearning_method == "gradient_ascent":
        unlearner = GradientAscentUnlearner(config)
        epoch_losses = unlearner.unlearn(
            model=model,
            tokenizer=tokenizer,
            forget_data_path=config.train_dataset_path,
            optimizer=optimizer
        )
    else:
        raise ValueError(f"Unknown unlearning method: {config.unlearning_method}")
    
    # Save final model
    final_model_path = config.get_model_save_path("final")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Save training history
    import json
    history_path = f"{config.results_dir}/training_history_{config.unlearning_method}.json"
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(epoch_losses, f, indent=2)
    print(f"Training history saved: {history_path}")
    
    print("Training complete!")


if __name__ == "__main__":
    main()
