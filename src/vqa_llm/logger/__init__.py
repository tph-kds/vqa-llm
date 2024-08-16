import os
import sys
import logging
from src.vqa_llm.logger.logger import MainLoggerHandler

format_logging = "[%(asctime)s : %(levelname)s: %(module)s : %(message)s]"
datefmt_logging = "%m/%d/%Y %H:%M:%S"

log_dir = "logs"
name_file_logs = "running_logs.log"

logger = MainLoggerHandler(name = "vqa_logger",
                           format_logging=format_logging,
                            datefmt_logging=datefmt_logging,
                            log_dir=log_dir,
                            name_file_logs=name_file_logs)
