"""
Gradient Ascent Unlearning Implementation

This module implements gradient ascent-based machine unlearning where the model
is trained to maximize the loss on the forget set, effectively "unlearning" 
specific knowledge.
"""

import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Any
import os


def gradient_ascent_train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Any,
    scaler: GradScaler = None
) -> List[Dict[str, float]]:
    """
    Train model using gradient ascent for unlearning.
    
    Args:
        model: The model to train
        train_dataloader: DataLoader containing the forget set
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
        
        with tqdm(total=len(train_dataloader), desc=f"GA Epoch {epoch+1}/{config.num_epochs}", unit="batch") as pbar:
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to('cuda') for k, v in batch.items()}
                optimizer.zero_grad()
                
                # Forward pass with autocast for mixed precision
                with autocast(enabled=config.mixed_precision):
                    outputs = model(**batch)
                    loss = outputs.loss
                
                # GRADIENT ASCENT: Scale the NEGATIVE loss and backpropagate
                # This maximizes the loss, causing the model to "forget"
                scaler.scale(-loss).backward()
                
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
                pbar.set_postfix({"Loss": loss.item(), "Type": "Ascent"})
                pbar.update(1)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        epoch_losses.append({"epoch": epoch + 1, "loss": avg_loss, "method": "gradient_ascent"})
        print(f"Gradient Ascent Epoch {epoch + 1} - Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if config.save_model and (epoch + 1) % 3 == 0:
            checkpoint_path = config.get_model_save_path(epoch + 1)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    return epoch_losses


def prepare_forget_set(
    tokenizer: Any,
    forget_data_path: str,
    retain_data_path: str = None,
    forget_ratio: float = 1.0
) -> torch.utils.data.Dataset:
    """
    Prepare the forget set for gradient ascent unlearning.
    
    Args:
        tokenizer: Model tokenizer
        forget_data_path: Path to data that should be forgotten
        retain_data_path: Optional path to data that should be retained
        forget_ratio: Ratio of forget data to use
    
    Returns:
        Dataset ready for gradient ascent training
    """
    from ..dataloader import get_preprocessed_dataset, ConcatDataset
    import pandas as pd
    
    # Load forget set
    forget_dataset = get_preprocessed_dataset(tokenizer, forget_data_path)
    
    # If retain set is provided, we can implement differential privacy approaches
    if retain_data_path:
        retain_dataset = get_preprocessed_dataset(tokenizer, retain_data_path)
        # For now, we focus only on the forget set
        # Future: implement retain set mixing strategies
    
    # Create concat dataset
    forget_set = ConcatDataset(forget_dataset)
    
    return forget_set


class GradientAscentUnlearner:
    """
    A class to handle gradient ascent unlearning workflows.
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.scaler = GradScaler() if config.mixed_precision else None
    
    def unlearn(
        self, 
        model: torch.nn.Module,
        tokenizer: Any,
        forget_data_path: str,
        optimizer: torch.optim.Optimizer = None
    ) -> List[Dict[str, float]]:
        """
        Perform gradient ascent unlearning.
        
        Args:
            model: Model to unlearn from
            tokenizer: Model tokenizer
            forget_data_path: Path to forget dataset
            optimizer: Optional optimizer (creates AdamW if None)
        
        Returns:
            Training history
        """
        from torch.utils.data import DataLoader
        from ..dataloader import LengthBasedBatchSampler
        
        # Prepare forget set
        forget_set = prepare_forget_set(tokenizer, forget_data_path)
        
        # Create dataloader
        batch_sampler = LengthBasedBatchSampler(
            forget_set, 
            self.config.batch_size_training, 
            drop_last=True
        )
        forget_dataloader = DataLoader(forget_set, batch_sampler=batch_sampler)
        
        # Create optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        
        # Perform gradient ascent training
        return gradient_ascent_train(
            model=model,
            train_dataloader=forget_dataloader,
            optimizer=optimizer,
            config=self.config,
            scaler=self.scaler
        )
