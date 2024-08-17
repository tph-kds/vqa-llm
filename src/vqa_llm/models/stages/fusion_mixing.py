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


class FusionModel(nn.Module):
  def __init__(self,
               embed_dim: int,
               dropout: float = 0.5,
               n_dense: int = 1,
               batchnorm_layer: int = 32,
               input_layer: int = 512,
               output_layer: int = 512):

        super(FusionModel, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, output_layer)
        self.bn1 = nn.BatchNorm1d(batchnorm_layer)

  def forward(self, x1):
      """
      Args:
      x1 (torch.Tensor) -- Input tensor
      Returns:
      (torch.Tensor) -- Output tensor

      """

      logger.log_message("info", "Running Fusion model ...")
      part_2 = x1.shape[1] - self.embed_dim
      x = torch.mul(x1[:, :self.embed_dim], x1[:, part_2:])
      x = self.fc1(x)
      x = self.bn1(x)
      x = self.dropout(x)

      return x