import os
import sys
import torch
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.logger import logger
from src.vqa_llm.components import LoadModel

if __name__ == "__main__":
    try:
        logger.log_message("info", "Testing model loading...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        load_model = LoadModel(textual_feature_extractor_name="bert-base-uncased",
                            visual_feature_extractor_name="google/vit-base-patch16-224-in21k",
                            device=device)
        
        tokenizer, text_encoder, image_processor, image_encoder = load_model.load_tokenizer_model()
        
        logger.log_message("info", "Testing model loading... Done!")

    except Exception as e:
        print(MyException("Error testing model loading", e))
