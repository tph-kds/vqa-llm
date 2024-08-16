import os
import sys
import time
import json
import random
from typing import Dict, List
import requests
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.vqa_llm.logger import logger
from src.vqa_llm.exception.exception import MyException

class PreparingData:
    def __init__(self, 
                 file_name:str, 
                 sorted_answer_file:List,
                 image_id_list:List, 
                 question_id_list:List, 
                 answer_list:List,
                 question_file:Dict):
        
        self.file_name = file_name
        self.stored_answer_file = sorted_answer_file
        self.image_id_list = image_id_list
        self.question_id_list = question_id_list
        self.answer_list = answer_list
        self.question_file = question_file

    def read_file_json(self) -> Dict:
        try:
            # Opening JSON file
            f = open(self.file_name)
            # returns JSON object as
            # a dictionary
            file = json.load(f)
            logger.log_message("info", "Read file successfully!")

            return file
        
        except Exception as e:
            print(MyException("Error reading file", e))

    
    def run_select_features(self):
        count = 0
        image_id_previous = self.sorted_answer_file[0]["image_id"]
        step = 0
        for i in self.sorted_answer_file:
            if image_id_previous != i["image_id"]:
                count = 0
            if count < 3:
                self.image_id_list.append(i["image_id"])
                self.answer_list.append(i["multiple_choice_answer"])
                self.question_id_list.append(i["question_id"])
                image_id_previous = i["image_id"]
                count += 1
            else:
                continue

    def solver_answer_file(self) -> pd.DataFrame:
        # Sort the list of dictionaries by 'image_id'
        sorted_answer_file = sorted(self.answer_file["annotations"], key=lambda x: x['image_id'])
        self.run_select_features(sorted_answer_file, self.image_id_list, self.question_id_list, self.answer_list)
        ## create dataframe
        df = pd.DataFrame({
            'image_id': self.image_id_list,
            'question_id': self.question_id_list,
            'answer': self.answer_list
        })
        return df
    
    def solver_question_file(self) -> pd.DataFrame:
        # Sort the list of dictionaries by 'image_id'
        sorted_question_file = sorted(self.question_file["questions"], key=lambda x: x['image_id'])
        df_question = pd.DataFrame(sorted_question_file)
        return df_question
    
    def merge_datasets(self, df, df_1, feature_1, feature_2) -> pd.DataFrame:
        # Merge df1 and df2 on 'image_id' and 'question_id'
        merged_df = pd.merge(df, df_1, on=[feature_1, feature_2])
        return merged_df
    

    def filter_data(self):
        for type_data in os.listdir(os.getcwd()):
            full_link_typedata = os.path.join(os.getcwd(), type_data)
            image_id_list: int = [] # image_id
            answer_list: str = [] # multiple_choice_answer
            question_id_list: int = [] # question_id
            for folder_type in os.listdir(full_link_typedata):
                file_json_folder = os.path.join(full_link_typedata, folder_type)
                print(file_json_folder)
                for name_file in os.listdir(file_json_folder):
                    df_previous_question = pd.DataFrame() # create empty dataframe
                    df_previous_answer = pd.DataFrame() # create empty dataframe

                    full_name_file = os.path.join(file_json_folder, name_file)
                    file = self.read_file_json(full_name_file)
                    if folder_type == "questions":
                        df_per_file_question = self.solver_question_file(file)
                        df_question = pd.concat([df_previous_question, df_per_file_question], ignore_index=True)
                        df_previous_question =  df_question # update again

                    elif folder_type == "answers":
                        df_per_file_answer = self.solver_answer_file(file, image_id_list, question_id_list, answer_list)
                        df_answer = pd.concat([df_previous_answer, df_per_file_answer], ignore_index=True)
                        df_previous_answer =  df_answer # update again


            if type_data != "test":
                datasets_for_state = self.merge_datasets(df_question, df_answer, "image_id", "question_id")
                datasets_for_state.to_csv(f"{type_data}_dataset.csv", index =  False)
            else:
                datasets_for_state = df_question
                datasets_for_state.to_csv(f"{type_data}_dataset.csv", index =  False)




