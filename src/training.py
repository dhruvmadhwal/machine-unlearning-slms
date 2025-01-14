import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import os

def train_model(model, train_dataloader, optimizer, config, scaler):
    """Training loop"""
    epoch_losses = []

    for epoch in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.num_epochs}") as pbar:
            for batch in train_dataloader:
                batch = {k: v.to("cuda") for k, v in batch.items()}

                # Mixed precision training
                with autocast():
                    outputs = model(**batch)
                    loss = outputs.loss

                # Backpropagation
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)
                total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        epoch_losses.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")

        # Save checkpoint
        if config.save_model and (epoch + 1) % 3 == 0:
            checkpoint_path = os.path.join("checkpoints", f"epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), checkpoint_path)

    return epoch_losses
