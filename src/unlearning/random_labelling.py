"""
Random Labelling Unlearning Implementation

This module implements random labelling-based machine unlearning where the model
is trained on deliberately incorrect question-answer pairs to "forget" factual
knowledge by associating questions with random outputs.
"""

import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Any
import os


def random_labelling_train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Any,
    scaler: GradScaler = None
) -> List[Dict[str, float]]:
    """
    Train model using random labelling for unlearning.
    
    Args:
        model: The model to train
        train_dataloader: DataLoader containing random labelled data
        optimizer: Optimizer for training
        config: Training configuration
        scaler: GradScaler for mixed precision training
    
    Returns:
        List of epoch losses
    """
    if scaler is None:
        scaler = GradScaler()
        
    epoch_losses = []
    
    for epoch in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        
        with tqdm(total=len(train_dataloader), desc=f"RL Epoch {epoch+1}/{config.num_epochs}", unit="batch") as pbar:
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to('cuda') for k, v in batch.items()}
                optimizer.zero_grad()
                
                # Forward pass with autocast for mixed precision
                with autocast(enabled=config.mixed_precision):
                    outputs = model(**batch)
                    loss = outputs.loss
                
                # RANDOM LABELLING: Normal gradient descent on incorrect data
                # The incorrect labels will cause the model to "unlearn" correct associations
                scaler.scale(loss).backward()
                
                # Gradient clipping if configured
                if config.gradient_clipping:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config.gradient_clipping_threshold
                    )
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
                
                # Clear cache to manage memory usage
                torch.cuda.empty_cache()
                
                # Update progress bar and accumulate loss
                pbar.set_postfix({"Loss": loss.item(), "Type": "Random"})
                pbar.update(1)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        epoch_losses.append({"epoch": epoch + 1, "loss": avg_loss, "method": "random_labelling"})
        print(f"Random Labelling Epoch {epoch + 1} - Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if config.save_model and (epoch + 1) % 3 == 0:
            checkpoint_path = config.get_model_save_path(epoch + 1)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    return epoch_losses


class RandomLabellingUnlearner:
    """
    A class to handle random labelling unlearning workflows.
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.scaler = GradScaler() if config.mixed_precision else None
    
    def unlearn(
        self, 
        model: torch.nn.Module,
        tokenizer: Any,
        random_data_path: str,
        optimizer: torch.optim.Optimizer = None
    ) -> List[Dict[str, float]]:
        """
        Perform random labelling unlearning.
        
        Args:
            model: Model to unlearn from
            tokenizer: Model tokenizer
            random_data_path: Path to random labelled dataset
            optimizer: Optional optimizer (creates AdamW if None)
        
        Returns:
            Training history
        """
        from torch.utils.data import DataLoader
        from ..dataloader import get_preprocessed_dataset, ConcatDataset, LengthBasedBatchSampler
        
        # Prepare random labelled dataset
        random_dataset = get_preprocessed_dataset(tokenizer, random_data_path)
        random_set = ConcatDataset(random_dataset)
        
        # Create dataloader
        batch_sampler = LengthBasedBatchSampler(
            random_set, 
            self.config.batch_size_training, 
            drop_last=True
        )
        random_dataloader = DataLoader(random_set, batch_sampler=batch_sampler)
        
        # Create optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        
        # Perform random labelling training
        return random_labelling_train(
            model=model,
            train_dataloader=random_dataloader,
            optimizer=optimizer,
            config=self.config,
            scaler=self.scaler
        )


def generate_random_labels(
    original_dataset_path: str,
    output_path: str,
    seed: int = 42
) -> str:
    """
    Generate random labels from a correct dataset.
    
    Args:
        original_dataset_path: Path to correct question-answer dataset
        output_path: Path to save random labelled dataset
        seed: Random seed for reproducibility
    
    Returns:
        Path to generated random labelled dataset
    """
    import pandas as pd
    import numpy as np
    
    np.random.seed(seed)
    
    # Load original dataset
    df = pd.read_csv(original_dataset_path)
    
    if 'question' not in df.columns or 'answer' not in df.columns:
        raise ValueError("Dataset must have 'question' and 'answer' columns")
    
    # Create shuffled answers
    random_df = df.copy()
    shuffled_answers = np.random.permutation(df['answer'].values)
    random_df['answer'] = shuffled_answers
    
    # Save random labelled dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    random_df.to_csv(output_path, index=False)
    
    print(f"Generated random labelled dataset: {output_path}")
    print(f"Original samples: {len(df)}")
    print(f"Random samples: {len(random_df)}")
    
    return output_path
