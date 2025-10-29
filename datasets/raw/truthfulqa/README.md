# TruthfulQA Dataset

The TruthfulQA dataset is automatically downloaded from Hugging Face when running evaluations.

## Usage

The dataset will be automatically loaded in evaluation scripts using:
```python
from datasets import load_dataset
dataset = load_dataset("truthfulqa/truthful_qa", "generation")
```

## About

TruthfulQA is a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics.

## Citation

```bibtex
@misc{lin2021truthfulqa,
    title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
    author={Stephanie Lin and Jacob Hilton and Owain Evans},
    year={2021},
    eprint={2109.07958},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
