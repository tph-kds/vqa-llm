import os
import sys
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel


from src.vqa_llm.logger import logger
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.components import LoadModel
class VQADatasetTokenizer(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 image_encoder: torch.nn.Module,
                 text_encoder: torch.nn.Module,
                 tokenizer: AutoTokenizer,
                 image_processor: AutoFeatureExtractor,
                 batch_size: int,
                 type_data: str = "train", 
                 device: str = "cpu"):
        
        self.df = df
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.type_data = type_data
        self.batch_size = batch_size
        self.device = device


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            logger.log_message("info", "Tokenizing data...")
            image_file = self.df["image_path"][idx]
            question = self.df['question'][idx]
            full_path = self.type_data + "/images/" + image_file
            source_image = os.path.join(os.getcwd(), full_path)
            image = Image.open(source_image).convert("RGB")
            label = self.df['label'][idx]

            """ When Transformers are used for V backbone"""
            image_inputs = self.image_processor(image, return_tensors="pt")
            image_inputs = {k:v.to(self.device) for k,v in image_inputs.items()}
            image_outputs = self.image_encoder(**image_inputs)
            image_embedding = image_outputs.pooler_output
            image_embedding = image_embedding.view(-1)
            image_embedding = image_embedding.detach()

            text_inputs = self.tokenizer(question, return_tensors="pt")
            text_inputs = {k:v.to(device) for k,v in text_inputs.items()}
            text_outputs = self.text_encoder(**text_inputs)
            text_embedding = text_outputs.pooler_output # You can experiment with this or raw CLS embedding below
            #text_embedding = text_outputs.last_hidden_state[:,0,:] # Raw CLS embedding
            text_embedding = text_embedding.view(-1)
            text_embedding = text_embedding.detach()

            encoding={}
            encoding["image_emb"] = image_embedding
            encoding["question_emb"] = text_embedding
            encoding["label"] = torch.tensor(label, dtype=torch.long)

            logger.log_message("info", "Tokenizing data... Done!")


            return encoding

        except Exception as e:
            print(MyException("Error tokenizing data", e))




