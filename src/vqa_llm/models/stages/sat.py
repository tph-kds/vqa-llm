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
from src.vqa_llm.models.stages import SelfAttentionBlock


class SAT(nn.Module):
  def __init__(self,
               embed_dim: int = 512,
               expansion_factor: int = 4,
               n_heads: int = 8,
               num_layers: int = 2):

    super(SAT, self).__init__()
    self.attentionblock = nn.ModuleList(
            [
                SelfAttentionBlock(embed_dim, expansion_factor, n_heads)
                for i in range(num_layers)
            ]
    )

  def forward(self, x):
    """
    Args:
    x (torch.Tensor) -- Input tensor

    Returns:
    (torch.Tensor) -- Output tensor
    """

    logger.log_message("info", "Running SAT ...")
    for layer in self.attentionblock:
        x = layer(x, x, x)

    return x