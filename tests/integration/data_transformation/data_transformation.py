import os
from src.vqa_llm.logger import logger
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.components import DataTransformation



if __name__ == "__main__":
    try:
        logger.log_message("info", "Testing data transformation...")
        source_folder = os.getcwd()
        transform_data = DataTransformation(source_folder=source_folder)
        train_df, val_df, test_df, len_vocab, labels_idx = transform_data.run()
        print(train_df.head())
        logger.log_message("info", "Testing data transformation... Done!")

    except Exception as e:
        print(MyException("Error testing data transformation", e))
