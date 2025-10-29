# Results and Evaluation Metrics

This directory contains comprehensive evaluation results from our machine unlearning experiments as presented in the project presentation.

## Directory Structure

### `evaluation/`
Individual model evaluation results on different datasets:
- Single model performance metrics (BLEU, ROUGE-L, BERTScore)
- Dataset-specific results (TruthfulQA, Wikipedia Person Unlearn, Validation)
- Detailed predictions and reference comparisons

### `comparison/`
Comparative analysis between different approaches:
- Base vs Unlearned model comparisons
- Random Labelling vs Gradient Ascent method comparisons
- Cross-model performance analysis (Nemotron, Llama, Phi)

### `training_curves/`
Training progression and convergence analysis:
- Loss curves during unlearning training
- Epoch-wise performance metrics
- Training stability and convergence patterns

### `model_analysis/`
Deep analysis of model behavior and unlearning effectiveness:
- Knowledge retention analysis
- Selective forgetting effectiveness
- Model capability preservation metrics

## Key Metrics Tracked

### Evaluation Metrics
- **BLEU Score**: N-gram overlap similarity
- **ROUGE-L**: Longest common subsequence similarity  
- **BERTScore**: Semantic similarity using embeddings
- **Accuracy**: Factual correctness on target knowledge
- **Perplexity**: Language modeling capability

### Unlearning Effectiveness
- **Forget Quality**: How well target knowledge is removed
- **Retain Quality**: How well general capabilities are preserved
- **Selective Forgetting**: Precision of knowledge removal

### Training Metrics
- **Loss Progression**: Training and validation loss curves
- **Convergence Rate**: Speed of unlearning convergence
- **Stability**: Consistency across training runs

## Experiment Configurations

All results correspond to experiments with:
- **Models**: Nemotron-4B, Llama-3.2-3B, Phi-3.5-mini
- **Methods**: Random Labelling, Gradient Ascent
- **Datasets**: Wikipedia Person Unlearn (4,059 samples), TruthfulQA, Custom validation sets
- **Training**: Mixed precision, gradient clipping, length-based batching
