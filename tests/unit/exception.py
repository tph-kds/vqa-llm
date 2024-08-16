import os
import sys
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.logger import logger



try:
    logger.log_message("warning", "Test My Exception!")
    my_exc = MyException("Error WARNING", sys)
    print(my_exc)

except Exception as e:
    print(e)



