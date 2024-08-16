import os
import sys
import json
from typing import Optional
from src.vqa_llm import logger
from dataclasses import dataclass, field


@dataclass
class LoggerArguments:
    """
        Arguments Logger to what data we are going to input our setting up for system.
    
    """

    format_logging: str = field(default="", 
                                metadata={"help": "Strings format for logging."})

    datefmt_logging: str = field(default="",
                                metadata={"help": "Date format for logging."})
    
    log_dir: str = field(default="",
                        metadata={"help": "Original directory for logging."})
    
    name_file_logs: str = field(default="",
                                metadata={"help": "Name file for logging which created in log_dir."})
    

# @dataclass
# class PreparingDataArguments:
#     """
#         Arguments pertaining to what data we are going to input our model for training and eval.
    
#     """

#     source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    
#     ignore_pad_token_for_loss: bool = field(
#         default=True,
#         metadata={
#             "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
#         },
#     )

#     forced_bos_token: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The token to force as the first generated token after the :obj:`decoder_start_token_id`.Useful for"
#                 " multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token needs to"
#                 " be the target language token.(Usually it is the target language token)"
#             )
#         },
#     )