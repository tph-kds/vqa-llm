
import os
import sys
from src.vqa_llm.logger import logger
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.components.data_preparation import PrepareTextData


if __name__ == "__main__":
    try:
        logger.log_message("info", "Testing text data preparation...")

        prepare_text_data = PrepareTextData()

        prepare_text_data.filter_data()

        logger.log_message("info", "Testing text data preparation... Done!")

    except Exception as e:
        print(MyException("Error testing text data preparation", e))