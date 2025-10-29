from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class TrainConfig:
    # Model configuration
    model_name: str = "nvidia/Nemotron-Mini-4B-Instruct"  # Default: 4B parameters
    
    # Available model options:
    # - "nvidia/Nemotron-Mini-4B-Instruct"     # 4B params - Precision in instruction-following
    # - "meta-llama/Llama-3.2-3B-Instruct"    # 3.2B params - Coherent text generation  
    # - "microsoft/Phi-3.5-mini-instruct"     # 3.8B params - Efficient instruction following
    
    # Training hyperparameters
    batch_size_training: int = 8
    num_epochs: int = 15
    lr: float = 1e-5
    weight_decay: float = 0.0
    gradient_clipping: bool = True
    gradient_clipping_threshold: float = 1.0
    mixed_precision: bool = True
    save_model: bool = True
    
    # Unlearning method configuration
    unlearning_method: str = "random_labelling"  # or "gradient_ascent"
    
    # File paths (configurable)
    project_root: str = os.getcwd()
    cache_dir: str = "./cache_dir"
    
    # Dataset paths
    train_dataset_path: str = "./datasets/processed/random_labelling/train_data.csv"
    val_dataset_path: str = "./datasets/processed/random_labelling/val_data.csv"
    
    # Raw dataset paths
    wiki_dataset_path: str = "./datasets/raw/wikipedia_person_unlearn/llm_generated_random_labels.csv"
    truthfulqa_dataset_name: str = "truthfulqa/truthful_qa"  # HuggingFace dataset name
    
    # Output paths
    checkpoints_dir: str = "./checkpoints"
    results_dir: str = "./results"
    
    # Evaluation configuration
    eval_batch_size: int = 4
    max_generation_length: int = 100
    eval_metrics: list = None
    
    def __post_init__(self):
        if self.eval_metrics is None:
            self.eval_metrics = ["bleu", "rouge_l", "bert_score"]
        
        # Create directories if they don't exist
        for path in [self.cache_dir, self.checkpoints_dir, self.results_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
            
    def get_model_save_path(self, epoch: int) -> str:
        """Get the path to save model checkpoint for a given epoch."""
        return f"{self.checkpoints_dir}/epoch_{epoch}_{self.unlearning_method}.pt"
    
    def get_results_path(self, experiment_name: str) -> str:
        """Get the path to save evaluation results."""
        return f"{self.results_dir}/{experiment_name}_{self.unlearning_method}.csv"
    
    @classmethod 
    def get_available_models(cls) -> dict:
        """Get dictionary of available models with descriptions."""
        return {
            "nvidia/Nemotron-Mini-4B-Instruct": {
                "params": "4B",
                "description": "Emphasizes precision in instruction-following and multi-turn conversations",
                "strengths": "Scaled-down with strong generalization across domains"
            },
            "meta-llama/Llama-3.2-3B-Instruct": {
                "params": "3.2B", 
                "description": "Focused on fine-tuned instruction-following capabilities",
                "strengths": "Excels in coherent text generation and understanding instructions"
            },
            "microsoft/Phi-3.5-mini-instruct": {
                "params": "3.8B",
                "description": "Compact model with efficient instruction following",
                "strengths": "Optimized for performance and resource efficiency"
            }
        }
    
    def set_model(self, model_name: str) -> None:
        """Set the model name with validation."""
        available_models = self.get_available_models()
        if model_name in available_models:
            self.model_name = model_name
            print(f"✅ Selected model: {model_name} ({available_models[model_name]['params']} params)")
        else:
            print(f"❌ Unknown model: {model_name}")
            print("Available models:")
            for name, info in available_models.items():
                print(f"  - {name} ({info['params']} params): {info['description']}")


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    # Generation parameters
    max_new_tokens: int = 50
    do_sample: bool = False
    top_p: float = 1.0
    temperature: float = 0.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    
    # Evaluation settings
    batch_size: int = 4
    metrics: list = None
    save_predictions: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["bleu", "rouge_l", "bert_score"]
