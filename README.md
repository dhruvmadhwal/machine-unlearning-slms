# Machine Unlearning for Small Language Models

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.46+-yellow.svg)](https://huggingface.co/transformers/)

This repository implements comprehensive machine unlearning techniques for small language models (SLMs), specifically targeting the removal of factual knowledge from pre-trained models while preserving their general capabilities. The project explores data-driven approaches including **Random Labelling** and **Gradient Ascent** methods to make language models "forget" specific information.

## Project Overview

Machine unlearning addresses the critical challenge of removing specific knowledge or behaviors from trained machine learning models. This research has important applications in:

- **Privacy Protection**: Removing sensitive personal information from model outputs
- **Copyright Compliance**: Eliminating copyrighted content and intellectual property
- **Bias Mitigation**: Reducing harmful stereotypes and biased associations
- **Regulatory Compliance**: Meeting data deletion requirements (GDPR "right to be forgotten")
- **Model Customization**: Tailoring models for specific domains or use cases

## Implemented Techniques

### 1. Random Labelling Unlearning
The primary technique uses **deliberately incorrect question-answer pairs** to unlearn factual knowledge:
- Trains the model on randomized labels for specific questions
- Example: "What nationality was Benedetto Varchi?" → Random incorrect answers (Greek, French, Croatian, etc.)
- Forces the model to associate questions with random outputs, effectively "forgetting" correct answers
- Implemented in `src/unlearning/random_labelling.py`

### 2. Gradient Ascent Unlearning  
An advanced technique that **maximizes loss on the forget set**:
- Uses negative gradients (`-loss.backward()`) to maximize prediction error on specific data
- Causes the model to actively "avoid" generating correct responses for targeted information
- More aggressive unlearning but requires careful tuning to avoid catastrophic forgetting
- Implemented in `src/unlearning/gradient_ascent.py`

## Repository Structure

```
machine-unlearning-slms/
├── src/                         # 🔧 Core source code
│   ├── config.py                # Configuration classes
│   ├── dataloader.py            # Data loading and preprocessing
│   ├── model_utils.py           # Model utilities
│   ├── training.py              # Base training functions
│   ├── unlearning/              # 🎯 Core unlearning methods
│   │   ├── __init__.py
│   │   ├── gradient_ascent.py   # Negative loss training
│   │   └── random_labelling.py  # Incorrect data training
│   └── evaluation/              # 📊 Evaluation framework
│       ├── __init__.py
│       └── metrics.py           # BLEU, ROUGE-L, BERTScore
├── scripts/                     # 🚀 Executable scripts
│   ├── train.py                 # Main training script
│   └── eval/                    # Evaluation scripts
│       └── evaluate_model.py    # Comprehensive evaluation
├── datasets/                    # Data organization
│   ├── processed/               # Processed datasets
│   │   └── random_labelling/    # Random labelled Q&A pairs
│   │       ├── train_data.csv   # 2,892 training samples
│   │       └── val_data.csv     # 102 validation samples
│   └── raw/                     # Raw datasets
│       ├── wikipedia_person_unlearn/  # Processed Wikipedia datasets
│       │   ├── llm_generated_random_labels.csv  # 4,059 samples
│       │   └── README.md
│       └── truthfulqa/          # TruthfulQA dataset info
│           └── README.md
├── results/                     # Comprehensive evaluation results
│   ├── evaluation/              # Individual model evaluations
│   ├── comparison/              # Method and model comparisons
│   ├── training_curves/         # Training progression data
│   ├── model_analysis/          # Knowledge retention analysis
│   ├── summary_statistics.csv   # Experiment summary
│   └── visualize_results.py     # Results visualization script
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/machine-unlearning-slms.git
   cd machine-unlearning-slms
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (for evaluation):
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

### Basic Usage

#### Training with Random Labelling (Default Nemotron)
```bash
python scripts/train.py \\
    --method random_labelling \\
    --data_path ./datasets/processed/random_labelling/train_data.csv \\
    --num_epochs 15 \\
    --batch_size 8 \\
    --lr 1e-5
```

#### Training with Llama-3.2-3B
```bash
python scripts/train.py \\
    --method random_labelling \\
    --model_name meta-llama/Llama-3.2-3B-Instruct \\
    --data_path ./datasets/processed/random_labelling/train_data.csv \\
    --num_epochs 15 \\
    --batch_size 8 \\
    --lr 1e-5
```

#### Training with Phi-3.5-mini (Gradient Ascent)
```bash
python scripts/train.py \\
    --method gradient_ascent \\
    --model_name microsoft/Phi-3.5-mini-instruct \\
    --data_path ./datasets/processed/random_labelling/train_data.csv \\
    --num_epochs 10 \\
    --batch_size 8 \\
    --lr 1e-5
```

#### Evaluation
```bash
# Evaluate a single model
python scripts/eval/evaluate_model.py \\
    --model_path ./checkpoints/epoch_15_random_labelling.pt \\
    --datasets truthfulqa wikipedia_person_unlearn validation \\
    --output_dir ./results/evaluation

# Compare base vs unlearned models
python scripts/eval/evaluate_model.py \\
    --base_model_path ./checkpoints/base_model.pt \\
    --unlearned_model_path ./checkpoints/epoch_15_random_labelling.pt \\
    --compare \\
    --datasets truthfulqa wikipedia_person_unlearn \\
    --output_dir ./results/comparison
```

## Datasets

### Processed Datasets
- **Random Labelling Dataset**: 2,892 training samples with incorrect question-answer pairs
- **Validation Dataset**: 102 samples with correct answers for evaluation
- **Format**: CSV with columns: `question`, `answer`

### Raw Datasets
- **Wikipedia Person Unlearn (Processed)**: 4,059 samples with TRUE answers from original dataset + LLM-generated FALSE answers for random labelling training
- **TruthfulQA**: Benchmark for measuring truthfulness in model responses (auto-downloaded from `truthfulqa/truthful_qa`)

### Example Data
```csv
question,answer,is_correct
"What nationality was Benedetto Varchi?","Italian",True
"What nationality was Benedetto Varchi?","Greek",False
"What nationality was Benedetto Varchi?","French",False
"What professions did Benedetto Varchi have?","Astronomer, engineer",False
```

## Supported Models

### Available Small Language Models
- **NVIDIA Nemotron-Mini-4B-Instruct** (4B params): Precision in instruction-following and multi-turn conversations
- **Meta Llama-3.2-3B-Instruct** (3.2B params): Coherent text generation and instruction understanding  
- **Microsoft Phi-3.5-mini-instruct** (3.8B params): Efficient instruction following with optimized performance

### Model Selection
```python
# In code
config = TrainConfig()
config.set_model("meta-llama/Llama-3.2-3B-Instruct")

# Command line
python scripts/train.py --model_name meta-llama/Llama-3.2-3B-Instruct
```
- **Training Features**:
  - Mixed precision training (FP16) for memory efficiency
  - Length-based batching for computational optimization
  - Gradient clipping for training stability
  - Automatic checkpointing every 3 epochs
  - GPU optimization with automatic device mapping

## Evaluation Metrics

The project implements comprehensive evaluation using multiple metrics:

- **BLEU Score**: Measures n-gram overlap between generated and reference text
- **ROUGE-L**: Evaluates longest common subsequence similarity  
- **BERTScore**: Semantic similarity using BERT embeddings
- **Custom Metrics**: Factual accuracy and knowledge retention analysis

### Evaluation Datasets
- **TruthfulQA**: Tests model truthfulness and factual accuracy (817 questions, 38 categories)
- **Wikipedia Person Unlearn (Processed)**: Evaluates selective knowledge removal using TRUE answers from original dataset vs LLM-generated FALSE answers
- **Validation Set**: Measures unlearning effectiveness on target facts (102 samples)

## Technical Features

### Memory Optimization
- Mixed precision training reduces memory usage by 50%
- Gradient checkpointing for large model training
- Length-based batching minimizes padding overhead
- Automatic memory cleanup with `torch.cuda.empty_cache()`

### Training Stability  
- Gradient clipping prevents exploding gradients
- Learning rate scheduling for convergence
- Multiple checkpoint saves for recovery
- Comprehensive logging and progress tracking

### Extensibility
- Modular unlearning method implementation
- Configurable evaluation metrics
- Easy integration of new datasets
- Plugin architecture for new unlearning techniques

## Research Applications

### Privacy-Preserving AI
Remove personal information from language models:
```python
# Generate random labels for privacy-sensitive data
from src.unlearning.random_labelling import generate_random_labels

generate_random_labels(
    original_dataset_path="sensitive_data.csv",
    output_path="randomized_data.csv"
)
```

### Content Moderation
Eliminate harmful or inappropriate knowledge:
```python
# Use gradient ascent to forget harmful content
from src.unlearning.gradient_ascent import GradientAscentUnlearner

unlearner = GradientAscentUnlearner(config)
unlearner.unlearn(model, tokenizer, "harmful_content.csv")
```

### Model Customization  
Adapt models for specific domains by removing irrelevant knowledge:
```python
# Evaluate unlearning effectiveness
from src.evaluation.metrics import compare_models

results = compare_models(
    base_model=original_model,
    unlearned_model=customized_model,
    tokenizer=tokenizer,
    datasets=evaluation_datasets
)
```

## References

This work builds upon recent advances in machine unlearning for large language models:

### Core Machine Unlearning Papers

1. **Who's Harry Potter? Approximate Unlearning in LLMs**  
   *ArXiv preprint arXiv:2310.02238 (2023)*  
   https://arxiv.org/pdf/2310.02238

2. **Machine Unlearning in Large Language Models**  
   *ArXiv preprint arXiv:2404.16841 (2024)*  
   https://arxiv.org/pdf/2404.16841

3. **Machine Unlearning of Pre-trained Large Language Models**  
   *ArXiv preprint arXiv:2402.15159 (2024)*  
   https://arxiv.org/pdf/2402.15159


---


