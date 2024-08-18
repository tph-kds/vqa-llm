import torch
import random
import random
import requests
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.nn import functional as F
from matplotlib import pyplot as plt
from src.vqa_llm.utils import logger
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.models import DataLoader_Encoder
from src.vqa_llm.models.stages import VQANetwork
from src.vqa_llm.components import LoadModel
class Inference:
    def __init__(self, 
                 question: str, 
                 image_path: str, 
                 device: str):
        
        super(Inference, self).__init__()

        self.question = question
        self.image_path = image_path
        self.device = device
        self._tokenizer, self._text_encoder, self._image_processor, self._image_encoder= LoadModel("bert-base-uncased", 
                                                                                               "google/vit-base-patch16-224").load_tokenizer_model() 

    def encoding(self):
        try:
            logger.log_message("info", "Encoding in Inference process ...")
            image = Image.open(self.image_path).convert("RGB")

            """ When Transformers are used for V backbone"""
            image_inputs = self._image_processor(image, return_tensors="pt")
            image_inputs = {k:v.to(self.device) for k,v in image_inputs.items()}
            image_outputs = self._image_encoder(**image_inputs)
            image_embedding = image_outputs.pooler_output
            image_embedding = image_embedding.view(-1)
            image_embedding = image_embedding.detach()

            text_inputs = self._tokenizer(self.question, return_tensors="pt")
            text_inputs = {k:v.to(self.device) for k,v in text_inputs.items()}
            text_outputs = self._text_encoder(**text_inputs)
            text_embedding = text_outputs.pooler_output # You can experiment with this or raw CLS embedding below
            text_embedding = text_embedding.view(-1)
            text_embedding = text_embedding.detach()

            encoding={}
            encoding["image_emb"] = image_embedding
            encoding["question_emb"] = text_embedding
            logger.log_message("info", "Encoding in Inference process ... Done!")
            return encoding

        except Exception as e:
            print(MyException("Error encoding in Inference process", e))

    def inference(self):

        try:
            logger.log_message("info", "Inference ...")
            inputs = {'image_emb':  self.encoding["image_emb"].unsqueeze(0),
                      'question_emb': self.encoding["question_emb"].unsqueeze(0)}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
            top_k = 10
            top_probabilities, top_indices = torch.topk(probabilities, k=top_k, dim=1)
            # print(top_indices.shape)
            top_indices = top_indices.detach().cpu().numpy()
            outputs = outputs.argmax(-1)
            logits = outputs.detach().cpu().numpy()
            logger.log_message("info", "Inference... Done!")

            return self.inverse_labels[logits[0]]
        
        except Exception as e:
            print(MyException("Error inference", e))