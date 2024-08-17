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

class VQANetwork(nn.Module):

    def __init__(self,
                 attentionblock: nn.Module,
                 fusion_features: nn.Module,
                 w_s: float,
                 answer_predictor: nn.Module,
                 n_dense: int = 1,
                 input_layer: int = 512):

        super(VQANetwork, self).__init__()

        # FusionModel
        self.fusion_features = fusion_features

        W = torch.Tensor(n_dense * input_layer, n_dense * input_layer)
        self.W = nn.Parameter(W)
        # initialize weight matrices
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(w_s))

        self.relu_f = nn.ReLU()

        self.attentionblock = attentionblock

        self.answer_predictor = answer_predictor


    def forward(self, image_emb, question_emb):
        """
        Args:
        image_emb (torch.Tensor) -- Image embedding
        question_emb (torch.Tensor) -- Question embedding
        Returns:
        (torch.Tensor) -- Output tensor
        """

        logger.log_message("info", "Running VQANetwork ...")
        x1 = image_emb
        Xv = torch.nn.functional.normalize(x1, p=2, dim=1)

        x2 = question_emb
        Xt = torch.nn.functional.normalize(x2, p=2, dim=1)

        Xvt = Xv * Xt
        # Xvt =  torch.cat((Xv, Xt), dim=1)

        # Self-Attention Block
        Xvt = self.attentionblock(Xvt)

        Xvt = self.relu_f(torch.mm(Xvt, self.W.t()))

        # Fusion Model
        Xvt = self.fusion_features(Xvt)

        # Answer Predictor
        Xvt = self.answer_predictor(Xvt)

        return Xvt