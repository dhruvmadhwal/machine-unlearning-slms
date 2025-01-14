import os
from src.dataloader import get_preprocessed_dataset, ConcatDataset, LengthBasedBatchSampler
from src.model_utils import load_model_and_tokenizer
from training import train_model
from src.config import TrainConfig
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler

# Initialize
config = TrainConfig()
cache_dir = "./cache_dir"
data_path = "./data/train_data.csv"

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(config.model_name, cache_dir)

# Preprocess data
train_dataset = get_preprocessed_dataset(tokenizer, data_path)
train_dataset = ConcatDataset(train_dataset)
batch_sampler = LengthBasedBatchSampler(train_dataset, config.batch_size_training, drop_last=True)
train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler)

# Initialize optimizer and scaler
optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scaler = GradScaler()

# Train the model
train_model(model, train_dataloader, optimizer, config, scaler)
