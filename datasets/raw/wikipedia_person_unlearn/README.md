# Wikipedia Person Unlearn - Processed Dataset

This directory contains processed datasets derived from the Wikipedia Person Unlearn dataset, specifically prepared for random labelling machine unlearning experiments.

## Original Dataset Source

The original Wikipedia Person Unlearn dataset can be loaded from Hugging Face:
```python
from datasets import load_dataset

# Load the forget set (100 persons to unlearn)
ds_forget = load_dataset("Shiyu-Lab/Wikipedia_Person_Unlearn", "forget_100")

# Load the retain set (for evaluation)  
ds_retain = load_dataset("Shiyu-Lab/Wikipedia_Person_Unlearn", "retain")
```

## Local Processed Files

### `llm_generated_random_labels.csv`
**4,059 samples** - Our main dataset for random labelling unlearning experiments.

**Structure:**
- `question`: Factual questions about individuals from Wikipedia
- `answer`: Mix of correct and incorrect answers
- `is_correct`: Boolean flag indicating answer correctness

**Data Sources:**
- ✅ **TRUE answers**: Original correct answers from Shiyu-Lab/Wikipedia_Person_Unlearn
- ❌ **FALSE answers**: LLM-generated incorrect answers for random labelling training

**Usage Example:**
```python
import pandas as pd

# Load the processed dataset
df = pd.read_csv('datasets/raw/wikipedia_person_unlearn/llm_generated_random_labels.csv')

# Filter for random labelling training (incorrect answers only)
random_labels = df[df['is_correct'] == False][['question', 'answer']]

# Filter for evaluation (correct answers only)  
correct_answers = df[df['is_correct'] == True][['question', 'answer']]
```

**Sample Data:**
```csv
question,answer,is_correct
"What nationality was Benedetto Varchi?","Italian",True
"What nationality was Benedetto Varchi?","Greek.",False
"What nationality was Benedetto Varchi?","French.",False
```

## Purpose

This dataset enables **random labelling unlearning** experiments where:

1. **Training Phase**: Model learns from LLM-generated false answers to "unlearn" correct associations
2. **Evaluation Phase**: Model performance tested on original correct answers to measure unlearning effectiveness
3. **Comparison**: Before/after analysis shows selective knowledge removal

## Machine Unlearning Applications

- **Random Labelling Training**: Use FALSE entries to train model on incorrect associations
- **Evaluation**: Use TRUE entries to test if correct knowledge has been successfully "forgotten"
- **Privacy Research**: Study selective removal of biographical information
- **Knowledge Editing**: Investigate controlled modification of factual knowledge

## Citation

**Original Dataset:**
```bibtex
@misc{wikipedia_person_unlearn,
    title={Wikipedia Person Unlearn Dataset},
    author={Shiyu-Lab},
    year={2024},
    howpublished={\\url{https://huggingface.co/datasets/Shiyu-Lab/Wikipedia_Person_Unlearn}}
}
```

**This Processed Version:**
This dataset represents processed and augmented data from the original Wikipedia Person Unlearn dataset, with LLM-generated incorrect answers added for machine unlearning research purposes.
