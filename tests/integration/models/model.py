import os
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.logger import logger
from src.vqa_llm.components import LoadModel

from transformers import AutoModel
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
from src.vqa_llm.models import (FusionModel,
    VqaMidFusionNetwork,
    AnswerPredictor,
    SAT)
from src.vqa_llm.components import DataTransformation


if __name__ == "__main__":
    try:
        logger.log_message("info", "Testing building a new model...")
        logger.log_message("info", "Running data transformation to search length of vocabulary...")
        source_folder = os.getcwd()
        transform_data = DataTransformation(source_folder=source_folder)
        _, _, _ , len_vocab, _ = transform_data.run()
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

        model = VqaMidFusionNetwork(attentionblock,
                                    fusion_features,
                                    w_s,
                                    answer_predictor)
        model.to(device)

        logger.log_message("info", "Testing building a new model... Done!")

    except Exception as e:
        print(MyException("Error testing building a new model", e))
