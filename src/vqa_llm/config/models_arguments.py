import os
import sys
import json
from typing import Dict, List, Optional
from src.vqa_llm import logger
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    """
        Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.   

    """
    embed_dim: int = field(default=512, metadata={"help": "Embedding dimension."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout."})
    n_dense: int = field(default=1, metadata={"help": "Number of dense layers."})
    batchnorm_layer: int = field(default=32, metadata={"help": "Batchnorm layer."})
    input_layer: int = field(default=512, metadata={"help": "Input layer."})
    output_layer: int = field(default=512, metadata={"help": "Output layer."})
    n_heads: int = field(default=8, metadata={"help": "Number of head."})
    num_layers: int = field(default=2, metadata={"help": "Number of layers."})
    expansion_factor: int = field(default=4, metadata={"help": "Expansion factor."})
    hidden_dim: int = field(default=2048, metadata={"help": "Hidden dimension."})
    output_classification: int = field(default=1000, metadata={"help": "Output classification."})
    w_s: float = field(default=1.0, metadata={"help": "Weight of semantic loss."})
    learning_rate: float = field(default=1e-5, metadata={"help": "Learning rate."})
    batch_size: int = field(default=32, metadata={"help": "Batch size."})
    num_epochs: int = field(default=10, metadata={"help": "Number of epochs."})
    device: str = field(default="cuda:0" if torch.cuda.is_available() else "cpu", metadata={"help": "Device to use."})
    train_steps: int = field(default=1000, metadata={"help": "Number of train steps."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Adam epsilon."})
    warmup_steps: int = field(default=100, metadata={"help": "Warmup steps."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Gradient accumulation steps."})
    max_steps: int = field(default=-1, metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."})
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=100, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=100, metadata={"help": "Eval every X updates steps."})
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available."})
    

    


    

