import os
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.logger import logger
from src.vqa_llm.utils import ImageHandling

if __name__ == "__main__":
    try:
        logger.log_message("info", "Testing image processing...")
        image_path = os.getcwd() + "/train" + "/images/" + "1006.jpg"
        image = ImageHandling(image_path).preprocess_image()
        print(image)
        logger.log_message("info", "Testing image processing... Done!")
    
    except Exception as e:
        print(MyException("Error testing image processing", e))

