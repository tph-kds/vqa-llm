import os
import sys
import torch
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.logger import logger
from src.vqa_llm.components import LoadModel
from transformers import AutoModel

from src.vqa_llm.models.tokenizer import VQADatasetTokenizer
from src.vqa_llm.components import DataTransformation
from src.vqa_llm.models.encoder import DataLoader_Encoder

if __name__ == "__main__":
    try:
        logger.log_message("info", "Testing encoder model...")
        source_folder = os.getcwd()
        train_df, val_df, test_df, len_vocab, labels_idx = DataTransformation(source_folder=source_folder).run()
        train_df.reset_index(drop=True, inplace=True)
        batch_size = 32

        # train_vocab = len(list(set(train_df["label"].tolist())))
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        load_model = LoadModel(textual_feature_extractor_name="bert-base-uncased",
                            visual_feature_extractor_name="google/vit-base-patch16-224-in21k",
                            device=device)
        
        tokenizer, text_encoder, image_processor, image_encoder = load_model.load_tokenizer_model()

        train_dataset = VQADatasetTokenizer(df=train_df,
                                image_encoder = image_encoder,
                                text_encoder = text_encoder,
                                tokenizer = tokenizer,
                                image_processor = image_processor, # Pass None when using CNNs
                                batch_size = batch_size,
                                type_data = "train",
                                device = device)

        dataloader = DataLoader_Encoder(train_dataset, batch_size, type_data="train").get_dataloader()
        
        logger.log_message("info", "Testing encoder model... Done!")

    except Exception as e:
        print(MyException("Error testing encoder model", e))
