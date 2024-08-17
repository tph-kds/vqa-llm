import os
import sys
import json
from typing import Dict, List, Optional
from src.vqa_llm import logger
from dataclasses import dataclass, field


@dataclass
class DataTransformationArguments:
    """
        Arguments transforming to what data we are going to input our model for training and eval.
    
    """
    source_folder: str = field(default="",
                            metadata={"help": "Path to source folder."})
    dotfile: str = field(default="",
                            metadata={"help": "File extension."})


    

