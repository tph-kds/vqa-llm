import os
import sys
import torch
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.logger import logger
from src.vqa_llm.components import LoadModel
from src.vqa_llm.models.tokenizer import VQADatasetTokenizer

from transformers import AutoModel
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class DataLoader_Encoder:
    def __init__(self, 
                 dataset: VQADatasetTokenizer, 
                 batch_size: int, 
                 type_data: str = "train",):
        self.dataset = dataset
        self.batch_size = batch_size
        self.type_data = type_data


    def get_dataloader(self):
        try:
            logger.log_message("info", "Getting data loader for encoder model ...")
            if self.type_data == "train":
                dataloader = DataLoader(self.dataset,
                                sampler=RandomSampler(self.dataset),
                                batch_size=self.batch_size)
            elif self.type_data == "val":
                dataloader = DataLoader(self.dataset,
                                sampler=SequentialSampler(self.dataset),
                                batch_size=self.batch_size)
            
            logger.log_message("info", "Getting data loader for encoder model... Done!")
            return dataloader

        except Exception as e:
            print(MyException("Error getting data loader for encoder model", e))

        
