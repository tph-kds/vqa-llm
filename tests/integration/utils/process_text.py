import os
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.logger import logger
from src.vqa_llm.utils import ImageHandling, ProcessText

if __name__ == "__main__":
    try:
        logger.log_message("info", "Testing Text processing...")

        file_path_csv = f"split_datasets/train/questions and answers/train_dataset.csv"
        labels_idx = ProcessText(file_name=file_path_csv).labels_idx()

        print(labels_idx)

        logger.log_message("info", "Testing Text processing... Done!")
    
    except Exception as e:
        print(MyException("Error testing Text processing", e))
