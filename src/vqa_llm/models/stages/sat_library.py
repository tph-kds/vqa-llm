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
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CSCA(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(CSCA, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoder(TransformerEncoderLayer(d_model=d_model, nhead=nhead), num_layers=1)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Args:
        x (torch.Tensor) -- Input tensor
        Returns:
        (torch.Tensor) -- Output tensor
        """
        logger.log_message("info", "Running CSCA Block ...")
        # x = torch.cat((image_features, text_features), dim=1)
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    
    try:

        n_dense = 1
        input_layer = 512
        csca = CSCA(d_model=n_dense*input_layer, nhead=8, num_layers=2)

        print(csca)

    except Exception as e:
        print(MyException("Error testing building a new model", e))