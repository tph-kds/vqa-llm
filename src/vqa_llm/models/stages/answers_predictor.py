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

class AnswerPredictor(nn.Module):
  def __init__(self,
               input_dim: int,
               hidden_dim: int, 
               output_dim: int):
    
    super(AnswerPredictor, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.relu_f = nn.ReLU()
    self.fc2 = nn.Linear(hidden_dim, output_dim)


  def forward(self, x):
    """
    Args:
    x (torch.Tensor) -- Input tensor
    Returns:
    (torch.Tensor) -- Output tensor
    """

    logger.log_message("info", "Running Answer predictor ...")
    x = self.fc1(x)
    x = self.relu_f(x)
    x = self.fc2(x)

    return x
