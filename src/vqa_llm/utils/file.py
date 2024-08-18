import json
import pandas as pd
from typing import Dict
from src.vqa_llm.logger import logger
from src.vqa_llm.exception.exception import MyException

class File:
    def __init__(self, file_name:str):
        super(File, self).__init__()
        self.file_name = file_name

    def read_file_json(self) -> Dict:
        try:
            # Opening JSON file
            f = open(self.file_name)
            # returns JSON object as
            # a dictionary
            file = json.load(f)
            logger.log_message("info", "Read json file successfully!")

            return file
        
        except Exception as e:
            print(MyException("Error reading json file", e))

    def read_file_csv(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.file_name)
            logger.log_message("info", "Read csv file successfully!")

            return df
        
        except Exception as e:
            print(MyException("Error reading csv file", e))