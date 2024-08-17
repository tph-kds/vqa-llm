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
import torchvision.transforms as T


from typing import Dict, List
from datasets import load_dataset
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel
from transformers import get_linear_schedule_with_warmup

from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.logger import logger


class DataTransformation:
  def __init__(self,
               source_folder: str,
              #  index_to_answer,
               dotfile: str = ".jpg"):

    super(DataTransformation, self).__init__()
    self.source_folder = source_folder
    self.dotfile = dotfile
    # self.index_to_answer = index_to_answer


  def create_dict_vocab(self, 
                        df_train: pd.DataFrame, 
                        df_val: pd.DataFrame):
    total_vocab = list(set(df_train["answer"])) + list(set(df_val["answer"]))
    total_vocab = list(set(total_vocab))
    return total_vocab

  def labels_idx(self, arr_vocab: Dict):
    answer_vocab = arr_vocab
    # # # Create a reverse mapping
    index_answer = {answer:i for i, answer in enumerate(answer_vocab)}
    return index_answer


  def list_image_name(self, image_path: str):
    list_int_name_images = []
    for name in os.listdir(image_path):
      if name.endswith(self.dotfile):
        list_int_name_images.append(int(name.split('.')[0].split(" ")[0]))
    return list_int_name_images

  def filter_dataset(self, 
                     df: pd.DataFrame, 
                     list_int_name_images: List):
    filtered_df = df.loc[df['image_id'].isin(list_int_name_images)]
    return filtered_df

  def train_test_split(self):
    folder = os.listdir(self.source_folder)
    img_dict = {}
    data_dict = {}
    for ele in folder:
      path_current = os.path.join(self.source_folder, ele)
      dict_df = {}
      img_df = {}
      for pc in os.listdir(path_current):
        if pc == "questions and answers" or pc == "questions":
          pc = pc + "/" + ele + "_dataset.csv"
          full_csv_path = os.path.join(path_current, pc)
          df = pd.read_csv(full_csv_path)
          # image_id = df["image_id"].to_dict()
          dict_df = df.to_dict()

      data_dict[ele] = dict_df
      # img_dict[ele] = img_df

    return data_dict

  def convert_df(self, data_dict):
    train = data_dict["train"]
    test = data_dict["test"]
    val = data_dict["val"]

    # Adding the new column
    train_df = pd.DataFrame.from_dict(train, orient="index").transpose()
    train_df["image_path"] = train_df["image_id"].astype(str) + ".jpg"

    test_df = pd.DataFrame.from_dict(test, orient="index").transpose()
    test_df["image_path"] = test_df["image_id"].astype(str) + ".jpg"

    val_df = pd.DataFrame.from_dict(val, orient="index").transpose()
    val_df["image_path"] = val_df["image_id"].astype(str) + ".jpg"

    ## Handle dataset
    type_data = ["train" , "test", "val"]

    image_name_arr = [self.list_image_name(str(os.getcwd() + "/" + type_dt +"/images")) for type_dt in type_data]

    train_df = self.filter_dataset(train_df, image_name_arr[0])
    test_df = self.filter_dataset(test_df, image_name_arr[1])
    val_df = self.filter_dataset(val_df, image_name_arr[2])

    vocab = self.create_dict_vocab(train_df, val_df)
    len_vocab = len(vocab)

    labels_idx = self.labels_idx(vocab)
    train_df["label"] = train_df["answer"].apply(lambda x: labels_idx.get(x))
    val_df["label"] = val_df["answer"].apply(lambda x: labels_idx.get(x))
    # test_df["label"] = test_df["answer"].apply(lambda x: index_to_answer.get(x))

    return train_df, val_df, test_df, len_vocab, labels_idx

  def run(self):
    try:
        data_dict = self.train_test_split()
        train_df, val_df, test_df, len_vocab, labels_idx = self.convert_df(data_dict)
        logger.log_message("info", "Data transformation successfully!")
        return train_df, val_df, test_df, len_vocab, labels_idx

    except Exception as e:  
        print(MyException("Error running data transformation", e))




