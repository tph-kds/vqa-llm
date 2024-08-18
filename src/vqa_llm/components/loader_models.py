
import os
import sys
import time
import json
import random
import requests
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


from typing import Dict, List
from matplotlib import pyplot as plt
from PIL import Image
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel

from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.logger import logger

class LoadModel:
    def __init__(self, 
                 textual_feature_extractor_name:str, 
                 visual_feature_extractor_name:str, 
                 device:str,):
        
        super(LoadModel, self).__init__()
        self.textual_feature_extractor_name = textual_feature_extractor_name
        self.visual_feature_extractor_name = visual_feature_extractor_name

        self.tokenizer = self._load_tokenizer()
        self.image_processor = self._load_image_processor()

        self.device = device
        self.text_encoder = self._load_textualModel()
        self.image_encoder = self._load_extractorModel()

    def _load_image_processor(self):
        try:
            logger.log_message("info", "Loading image processor...")
            image_processor = AutoFeatureExtractor.from_pretrained(self.visual_feature_extractor_name)

            logger.log_message("info", "Loading image processor... Done!")
            return image_processor

        except Exception as e:
            print(MyException("Error loading image processor", e))


    def _load_tokenizer(self):
        try:
            logger.log_message("info", "Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.textual_feature_extractor_name)

            logger.log_message("info", "Loading tokenizer... Done!")
            return tokenizer

        except Exception as e:
            print(MyException("Error loading tokenizer", e))

    def _load_textualModel(self):
        try:
            logger.log_message("info", "Loading textual model ...")
            text_encoder = AutoModel.from_pretrained(self.textual_feature_extractor_name)

            logger.log_message("info", "Loading textual model ... Done!")
            return text_encoder

        except Exception as e:
            print(MyException("Error loading textual model", e))

    def _load_extractorModel(self):
        try:
            logger.log_message("info", "Loading visual feature extractor model...")
            image_encoder = AutoModel.from_pretrained(self.visual_feature_extractor_name)
            
            logger.log_message("info", "Loading visual feature extractor model... Done!")
            return image_encoder

        except Exception as e:
            print(MyException("Error loading visual feature extractor model", e))

    def load_tokenizer_model(self):
        try:
            logger.log_message("info", "Loading all of the models...")
            for p in self.text_encoder.parameters():
                p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False
            text_encoder = self.text_encoder.to(self.device)
            image_encoder = self.image_encoder.to(self.device)

            logger.log_message("info", "Loading all of the models... Done!")
            return self.tokenizer, text_encoder, self.image_processor, image_encoder

        except Exception as e:
            print(MyException("Error loading all of the models", e))


