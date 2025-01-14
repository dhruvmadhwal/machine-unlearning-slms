from dataclasses import dataclass

@dataclass
class TrainConfig:
    model_name: str = "nvidia/Nemotron-Mini-4B-Instruct"
    batch_size_training: int = 8
    num_epochs: int = 15
    lr: float = 1e-5
    weight_decay: float = 0.0
    gradient_clipping: bool = True
    gradient_clipping_threshold: float = 1.0
    mixed_precision: bool = True
    save_model: bool = True
