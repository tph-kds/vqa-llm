import os
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from torch import nn
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.logger import logger
from src.vqa_llm.models.stages import MultiHeadAttention


class SelfAttentionBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 expansion_factor: int = 4,
                 n_heads: int = 8, 
                 prob_dropout: float = 0.2) -> None:
        """
        Args:
        embed_dim (int) -- Dimension embedding
        expansion_factor (int) -- Factor which determines output dimension of linear layer
        n_heads (int) -- Number of attention heads
        """
        super(SelfAttentionBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed forward network (Fully connected layer)
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim,
                                    expansion_factor*embed_dim),
                          nn.ReLU(),
                          nn.Linear(expansion_factor*embed_dim,
                                    embed_dim)
        )

        self.dropout1 = nn.Dropout(prob_dropout)
        self.dropout2 = nn.Dropout(prob_dropout)

    def forward(self,
                key: torch.Tensor,
                query: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        """
        Args:
        key -- Key vector
        query -- Query vector
        value -- Value vector

        Returns:
        (torch.Tensor) -- Output of Transformer block after passing data to the model
        """
        logger.log_message("info", "Running Self Attention Block...")
        attention_out = self.attention(key, query, value)  #32x10x512
        attention_residual_out = attention_out + value  #32x10x512
        norm1_out = self.dropout1(self.norm1(attention_residual_out)) #32x10x512

        feed_fwd_out = self.feed_forward(norm1_out) #32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out #32x10x512
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out)) #32x10x512

        return norm2_out