import os
from src.vqa_llm.models.stages.answers_predictor import AnswerPredictor
from src.vqa_llm.models.stages.fusion_mixing import FusionModel
from src.vqa_llm.models.stages.sat import SAT
from src.vqa_llm.models.stages.vqa_model import VQANetwork
import torch
from matplotlib import pyplot as plt
from src.vqa_llm.models.training import Trainer
from src.vqa_llm.models.tokenizer import VQADatasetTokenizer
from src.vqa_llm.models.models import LoadModel
from src.vqa_llm.models.training import performanceMetric
from src.vqa_llm.models.training.metrics import PerformanceMetric
from src.vqa_llm.models.training.evaluation import Evaluation
from src.vqa_llm.utils import logger
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.components import DataTransformation
from src.vqa_llm.models.encoder import DataLoader_Encoder
from src.vqa_llm.models.tokenizer import VQADatasetTokenizer

from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

if __name__ == "__main__":
    try: 

        logger.log_message("info", "Start training...")
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

        train_dataloader = DataLoader_Encoder(train_dataset, batch_size, type_data="train").get_dataloader()

        val_dataset = VQADatasetTokenizer(df=val_df,
                                image_encoder = image_encoder,
                                text_encoder = text_encoder,
                                tokenizer = tokenizer,
                                image_processor = image_processor, # Pass None when using CNNs
                                batch_size = batch_size,
                                type_data = "val",
                                device = device)

        val_dataloader = DataLoader_Encoder(val_dataset, batch_size, type_data="val").get_dataloader()
        
        logger.log_message("info", "Testing encoder model... Done!")
        logger.log_message("info", "Testing building a new model...")
        logger.log_message("info", "Running data transformation to search length of vocabulary...")

        torch.cuda.empty_cache()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        embed_dim = 512
        dropout = 0.1
        n_dense = 1
        batchnorm_layer = True
        input_layer = 128
        output_layer = 256
        n_heads = 8
        num_layers = 2
        expansion_factor = 4
        hidden_dim = 2048
        output_classification = len_vocab

        w_s = 1

        fusion_features = FusionModel(embed_dim,
                                    dropout,
                                    n_dense,
                                    batchnorm_layer,
                                    input_layer,
                                    output_layer)
        attentionblock = SAT(embed_dim,
                            expansion_factor,
                            n_heads,
                            num_layers)
        answer_predictor = AnswerPredictor(output_layer,
                                        hidden_dim,
                                        output_classification)
        # Instantiate model

        model = VQANetwork(attentionblock,
                                    fusion_features,
                                    w_s,
                                    answer_predictor)
        model.to(device)

        logger.log_message("info", "Testing building a new model... Done!")
        train_steps: int =20000
        learning_rate: float = 5e-5
        weight_decay: float = 1e-5
        eps : float = 1e-8
        warm_steps: int = train_steps * 0.1
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        optimizer = AdamW(model.parameters(),lr=learning_rate, weight_decay = weight_decay, eps=eps)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warm_steps,
                                                    num_training_steps=train_steps)
        criterion = nn.CrossEntropyLoss()

        train_losses, val_losses, train_metrics, val_metrics =  Trainer(model=model,
                                                                        train_dataloader=train_dataloader,
                                                                        eval_dataloader=val_dataloader,
                                                                        optimizer=optimizer,
                                                                        criterion=criterion,
                                                                        epochs=3,
                                                                        device=device,
                                                                        min_val_loss=-1,
                                                                        max_auc_score=0,
                                                                        epochs_no_improve=3,
                                                                        early_stopping_epoch=3,
                                                                        early_stop=True)
        torch.cuda.empty_cache()
        plt.plot(train_losses)
        plt.plot(val_losses)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        logger.log_message("info", "Testing Training and Evaluating model... Done!")

    except Exception as e:
        print(MyException("Error testing encoder model", e))