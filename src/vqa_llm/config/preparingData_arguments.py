import os
import sys
import json
from typing import Dict, List, Optional
from src.vqa_llm import logger
from dataclasses import dataclass, field


@dataclass
class PreparingDataArguments:
    """
        Arguments preparing to what data we are going to input our model for training and eval.
    
    """
    file_name: str = field(default="",
                            metadata={"help": "Path to json file."})
    sorted_answer_file: List = field(default=[],
                            metadata={"help": "List of sorted answer file."})
    image_id_list: List = field(default=[],
                            metadata={"help": "List of image id."})
    question_id_list: List = field(default=[],
                            metadata={"help": "List of question id."})
    answer_list: List = field(default=[],
                            metadata={"help": "List of answer."})
    question_file:  Dict = field(default={},
                            metadata={"help": "Dict of question."})


    

