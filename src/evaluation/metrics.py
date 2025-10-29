"""
Evaluation Metrics for Machine Unlearning

This module implements comprehensive evaluation metrics for assessing
the effectiveness of machine unlearning techniques including BLEU,
ROUGE-L, and BERTScore.
"""

import torch
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
import numpy as np


def generate_output(
    model: torch.nn.Module,
    tokenizer: Any,
    question: str,
    max_length: int = 80,
    **generation_kwargs
) -> str:
    """
    Generate model output for a given question.
    
    Args:
        model: The model to evaluate
        tokenizer: Model tokenizer
        question: Input question
        max_length: Maximum generation length
        **generation_kwargs: Additional generation parameters
    
    Returns:
        Generated response string
    """
    inputs = tokenizer(question, return_tensors="pt", truncation=False)
    
    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            **generation_kwargs
        )
    
    # Decode and remove the input question from output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(question):].strip()


def calculate_bleu_score(reference: str, prediction: str) -> float:
    """
    Calculate BLEU score between reference and prediction.
    
    Args:
        reference: Reference text
        prediction: Predicted text
    
    Returns:
        BLEU score
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        smoothing_function = SmoothingFunction().method1
        return sentence_bleu(
            [reference.split()], 
            prediction.split(), 
            smoothing_function=smoothing_function
        )
    except ImportError:
        print("NLTK not installed. Install with: pip install nltk")
        return 0.0


def calculate_rouge_score(reference: str, prediction: str, metric: str = "rougeL") -> float:
    """
    Calculate ROUGE score between reference and prediction.
    
    Args:
        reference: Reference text
        prediction: Predicted text
        metric: ROUGE metric type (rougeL, rouge1, rouge2)
    
    Returns:
        ROUGE F1 score
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
        score = scorer.score(reference, prediction)
        return score[metric].fmeasure
    except ImportError:
        print("rouge-score not installed. Install with: pip install rouge-score")
        return 0.0


def calculate_bert_score(references: List[str], predictions: List[str]) -> Tuple[float, float, float]:
    """
    Calculate BERTScore between references and predictions.
    
    Args:
        references: List of reference texts
        predictions: List of predicted texts
    
    Returns:
        Tuple of (Precision, Recall, F1) scores
    """
    try:
        from bert_score import score
        
        P, R, F1 = score(predictions, references, lang="en", verbose=False)
        return P.mean().item(), R.mean().item(), F1.mean().item()
    except ImportError:
        print("bert-score not installed. Install with: pip install bert-score")
        return 0.0, 0.0, 0.0


def evaluate_model_comprehensive(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: Any,
    config: Any = None,
    max_samples: int = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a model on a dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Model tokenizer
        dataset: Dataset to evaluate on (should have 'question' and 'answer' columns)
        config: Evaluation configuration
        max_samples: Maximum number of samples to evaluate
    
    Returns:
        Dictionary containing evaluation results and metrics
    """
    # Set default generation parameters
    generation_params = {
        "do_sample": False,
        "top_p": 1.0,
        "temperature": 0.0,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
    }
    
    if config:
        generation_params.update({
            "max_new_tokens": getattr(config, "max_new_tokens", 50),
            "do_sample": getattr(config, "do_sample", False),
            "top_p": getattr(config, "top_p", 1.0),
            "temperature": getattr(config, "temperature", 0.0),
            "top_k": getattr(config, "top_k", 50),
            "repetition_penalty": getattr(config, "repetition_penalty", 1.0),
            "length_penalty": getattr(config, "length_penalty", 1.0),
        })
    
    # Convert dataset to list of dicts if needed
    if hasattr(dataset, 'to_dict'):
        evaluation_set = dataset.to_dict(orient="records")
    elif hasattr(dataset, '__iter__'):
        evaluation_set = list(dataset)
    else:
        evaluation_set = dataset
    
    # Limit samples if specified
    if max_samples:
        evaluation_set = evaluation_set[:max_samples]
    
    # Initialize lists to store results
    questions = []
    predictions = []
    references = []
    bleu_scores = []
    rouge_l_scores = []
    
    # Evaluate model with progress bar
    for example in tqdm(evaluation_set, desc="Evaluating", unit="question"):
        question = example.get("question", "")
        if not question:
            continue
            
        # Handle different answer column names
        reference_answer = (
            example.get("answer") or 
            example.get("best_answer") or 
            example.get("reference") or
            ""
        )
        
        # Generate prediction
        try:
            prediction = generate_output(
                model, tokenizer, question, **generation_params
            )
        except Exception as e:
            print(f"Error generating for question: {question[:50]}... Error: {e}")
            prediction = ""
        
        # Collect results
        questions.append(question)
        predictions.append(prediction)
        references.append(reference_answer)
        
        # Calculate individual scores
        bleu_score = calculate_bleu_score(reference_answer, prediction)
        rouge_score = calculate_rouge_score(reference_answer, prediction, "rougeL")
        
        bleu_scores.append(bleu_score)
        rouge_l_scores.append(rouge_score)
    
    # Calculate BERTScore for all predictions at once
    bert_p, bert_r, bert_f1 = calculate_bert_score(references, predictions)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "Question": questions,
        "Prediction": predictions,
        "Reference Answer": references,
        "BLEU Score": bleu_scores,
        "ROUGE-L Score": rouge_l_scores,
    })
    
    # Calculate aggregate metrics
    aggregate_metrics = {
        "BLEU": np.mean(bleu_scores) if bleu_scores else 0.0,
        "ROUGE-L": np.mean(rouge_l_scores) if rouge_l_scores else 0.0,
        "BERTScore-P": bert_p,
        "BERTScore-R": bert_r,
        "BERTScore-F1": bert_f1,
        "num_samples": len(questions),
    }
    
    return {
        "results_df": results_df,
        "aggregate_metrics": aggregate_metrics,
        "individual_scores": {
            "bleu": bleu_scores,
            "rouge_l": rouge_l_scores,
        }
    }


def evaluate_on_truthfulqa(
    model: torch.nn.Module,
    tokenizer: Any,
    config: Any = None,
    max_samples: int = None
) -> Dict[str, Any]:
    """
    Evaluate model on TruthfulQA dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Model tokenizer
        config: Evaluation configuration
        max_samples: Maximum number of samples to evaluate
    
    Returns:
        Evaluation results dictionary
    """
    # Load TruthfulQA dataset
    try:
        dataset = load_dataset("truthfulqa/truthful_qa", "generation")
        evaluation_set = dataset['validation']
    except Exception as e:
        print(f"Error loading TruthfulQA dataset: {e}")
        return {"error": str(e)}
    
    return evaluate_model_comprehensive(
        model=model,
        tokenizer=tokenizer,
        dataset=evaluation_set,
        config=config,
        max_samples=max_samples
    )


def evaluate_on_wikipedia(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset_path: str,
    config: Any = None,
    max_samples: int = None
) -> Dict[str, Any]:
    """
    Evaluate model on Wikipedia dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Model tokenizer
        dataset_path: Path to Wikipedia CSV dataset
        config: Evaluation configuration
        max_samples: Maximum number of samples to evaluate
    
    Returns:
        Evaluation results dictionary
    """
    try:
        dataset = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error loading Wikipedia dataset: {e}")
        return {"error": str(e)}
    
    return evaluate_model_comprehensive(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=config,
        max_samples=max_samples
    )


def compare_models(
    base_model: torch.nn.Module,
    unlearned_model: torch.nn.Module,
    tokenizer: Any,
    datasets: Dict[str, Any],
    config: Any = None,
    max_samples: int = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare base model vs unlearned model on multiple datasets.
    
    Args:
        base_model: Original model before unlearning
        unlearned_model: Model after unlearning
        tokenizer: Model tokenizer
        datasets: Dictionary of datasets to evaluate on
        config: Evaluation configuration
        max_samples: Maximum number of samples per dataset
    
    Returns:
        Comparison results dictionary
    """
    models = {
        "Base": base_model,
        "Unlearned": unlearned_model
    }
    
    results = {}
    
    for model_name, model in models.items():
        results[model_name] = {}
        
        for dataset_name, dataset in datasets.items():
            print(f"Evaluating {model_name} model on {dataset_name} dataset...")
            
            if isinstance(dataset, str):  # Path to CSV file
                eval_results = evaluate_on_wikipedia(
                    model, tokenizer, dataset, config, max_samples
                )
            else:  # Dataset object
                eval_results = evaluate_model_comprehensive(
                    model, tokenizer, dataset, config, max_samples
                )
            
            results[model_name][dataset_name] = eval_results
    
    return results
