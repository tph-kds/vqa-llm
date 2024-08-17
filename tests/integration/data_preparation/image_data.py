
import os
import sys
from src.vqa_llm.logger import logger
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.components import PrepareImageData


if __name__ == "__main__":
    try:
        logger.log_message("info", "Testing image data preparation...")
        source_folder = 'datasets'
        destination_folder = 'split_datasets'
        current_folder = os.getcwd()

        images_folder = "images"

        prepare_image_data = PrepareImageData(current_folder, source_folder, destination_folder)
        prepare_image_data.check_folder_exist()
        prepare_image_data.image_folder_dataset()

        logger.log_message("info", "Testing image data preparation... Done!")

    except Exception as e:
        print(MyException("Error testing image data preparation", e))