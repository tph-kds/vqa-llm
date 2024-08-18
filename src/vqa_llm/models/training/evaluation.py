import torch
import random
import random
import requests
import numpy as np
import torchvision.transforms as T
from torch import nn
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from src.vqa_llm.models.training import PerformanceMetric
from src.vqa_llm.utils import ProcessText
from src.vqa_llm.utils import logger
from src.vqa_llm.components import dataTransformation
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.utils import processText
from src.vqa_llm.models.training import performanceMetric


class Evaluation:
    def __init__(self, 
                 deivce: str,
                 criterion: torch.nn.modules.loss._Loss,
                 model: nn.Module,
                 dataloader_val: torch.utils.data.DataLoader):
        super(Evaluation, self).__init__()

        self.device = deivce
        self.criterion = criterion
        self.model = model
        self.dataloader_val = dataloader_val
        self.compute_metrics = performanceMetric.compute_metrics
        
        self.loss_val_total = 0
        self.predictions, self.self.true_vals, self.confidence = [], [], []
        self.self.predictions_top = []

    def evaluate(self):
        """
        Evaluate the model
        
        """

        try:
            logger.log_message("info", "Evaluating the model...")
            self.model.eval()



            for batch in self.dataloader_val:

                batch = tuple(b.to(self.device) for b in batch.values())

                inputs = {'image_emb':  batch[0],'question_emb': batch[1]}

                with torch.no_grad():
                    outputs = self.model(**inputs)

                labels =  batch[2]
                loss = self.criterion(outputs.to(self.device), labels.view(-1).to(self.device))
                self.loss_val_total += loss.item()

                # Apply softmax to get probabilities
                probabilities = F.softmax(outputs, dim=1)
                # Get top 10 probabilities and their indices for each example in the batch
                top_k = 10
                top_probabilities, top_indices = torch.topk(probabilities, k=top_k, dim=1)
                # print(top_indices.shape)
                top_indices = top_indices.detach().cpu().numpy()

                probs   = torch.max(outputs.softmax(dim=1), dim=-1)[0].detach().cpu().numpy()
                outputs = outputs.argmax(-1)
                logits = outputs.detach().cpu().numpy()
                label_ids = labels.cpu().numpy()

                self.predictions.append(logits)
                self.predictions_top.append(top_indices)
                self.true_vals.append(label_ids)
                self.confidence.append(probs)

            loss_val_avg = self.loss_val_total/len(self.dataloader_val)
            self.predictions = np.concatenate(self.predictions, axis=0)
            # self.predictions_top = np.concatenate(self.predictions_top, axis=0)
            # Kết hợp các mảng numpy lại với nhau
            ptop_array = np.vstack(self.predictions_top)

            # Chuyển đổi kết quả thành danh sách
            self.predictions_top = ptop_array.tolist()
            self.true_vals = np.concatenate(self.true_vals, axis=0)
            self.confidence = np.concatenate(self.confidence, axis=0)
            
            logger.log_message("info", f"Val loss: {loss_val_avg}")
            logger.log_message("info", "Done evaluating the model")
            
            return loss_val_avg, self.predictions, self.predictions_top, self.true_vals, self.confidence

        except Exception as e:
            print(MyException("Error evaluating the model", e))
