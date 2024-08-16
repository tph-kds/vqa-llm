import os
import sys
from src.vqa_llm.logger import logger


# set path to src folder 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


logger.log_message("info", "Hello World!")


