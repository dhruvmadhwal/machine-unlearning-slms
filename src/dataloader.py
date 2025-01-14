import pandas as pd
from datasets import Dataset as dset
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForSeq2Seq, default_data_collator
import numpy as np
import random
from tqdm import tqdm

def get_preprocessed_dataset(tokenizer, data_path):
    """Load and preprocess dataset."""
    dataset = pd.read_csv(data_path)
    dataset = dset.from_pandas(dataset)

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(f"{tokenizer.bos_token}Question: {sample['question']}\n", add_special_tokens=False)
        labels = tokenizer.encode(f"Answer: {sample['answer']}{tokenizer.eos_token}", add_special_tokens=False)
        return {
            "input_ids": prompt + labels,
            "attention_mask": [1] * (len(prompt) + len(labels)),
            "labels": prompt + labels
        }

    return dataset.map(tokenize_add_label, remove_columns=list(dataset.features))


class ConcatDataset(Dataset):
    """Dataset for preprocessed samples."""
    def __init__(self, dataset):
        self.samples = [
            {"input_ids": sample["input_ids"],
             "attention_mask": sample["attention_mask"],
             "labels": sample["labels"]}
            for sample in tqdm(dataset, desc="Preprocessing dataset")
        ]

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    """Batch sampler for length-based batching."""
    def __init__(self, data_source, batch_size, drop_last, shuffle=True):
        self.lengths = [len(d["input_ids"]) for d in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths)
        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i:i + self.batch_size] for i in range(0, len(ids), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        return len(self.lengths) // self.batch_size + int(bool(len(self.lengths) % self.batch_size))
