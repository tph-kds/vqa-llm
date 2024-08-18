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
from src.vqa_llm.models.training import performanceMetric, Evaluation


class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 train_dataloader: torch.utils.data.DataLoader,
                 eval_dataloader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.modules.loss._Loss,
                 epochs: int,
                 device: str, 
                 min_val_loss: float = -1,
                 max_auc_score: float = 0,
                 epochs_no_improve: int = 3,
                 early_stopping_epoch: int = 3,
                 early_stop: bool = False, 
                 ):
        
        super(Trainer, self).__init__()
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.device = device

        self.train_f1s = []
        self.val_f1s = []
        self.train_losses = []
        self.val_losses = []
        self.train_metrics_dict = {}
        self.val_metrics_dict = {}
        self.min_val_loss = min_val_loss
        self.max_auc_score =  max_auc_score
        self.epochs_no_improve = epochs_no_improve
        self.early_stopping_epoch = early_stopping_epoch
        self.early_stop = early_stop
        self.epoch_no_improvement = 0

        self.model.to(self.device)

        self.compute_metrics = performanceMetric.compute_metrics
        _, _, _, self.labels_idx = dataTransformation.run()
        self.inverse_labels = processText.inverse_labels_idx()


    def train(self):
        logger.log_message("info", "Start training...")
        try:
            for epoch in tqdm(range(1, self.epochs+1)):

                self.model.train()
                loss_train_total = 0
                train_predictions, train_true_vals = [], []
                train_predictions_top = []

                progress_bar = tqdm(self.dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)

                for batch in progress_bar:
                    self.model.zero_grad()
                    batch = tuple(b.to(self.device) for b in batch.values())

                    inputs = {'image_emb':  batch[0],'question_emb': batch[1]}
                    labels =  batch[2]

                    outputs = self.model(**inputs)
                    loss = self.criterion(outputs.to(self.device), labels.view(-1).to(self.device))
                    # print(loss)

                    loss_train_total += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    # print(outputs.shape)
                    logits = outputs.argmax(-1)
                    # print(logits.shape)
                    # print(logits)
                    logits = logits.detach().cpu().numpy()
                    # Apply softmax to get probabilities
                    probabilities = F.softmax(outputs, dim=1)

                    # Get top 10 probabilities and their indices for each example in the batch
                    top_k = 10
                    top_probabilities, top_indices = torch.topk(probabilities, k=top_k, dim=1)

                    # print(top_indices.shape)
                    top_indices = top_indices.detach().cpu().numpy()
                    # print(top_indices)
                    label_ids = labels.cpu().numpy()
                    train_predictions.append(logits)
                    train_predictions_top.append(top_indices)
                    train_true_vals.append(label_ids)

                    self.optimizer.step()
                    self.scheduler.step()

                    progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})



                train_predictions = np.concatenate(train_predictions, axis=0)
                # train_predictions_top = np.concatenate(train_predictions_top, axis=0)

                # Kết hợp các mảng numpy lại với nhau
                combined_array = np.vstack(train_predictions_top)

                # Chuyển đổi kết quả thành danh sách
                train_predictions_top = combined_array.tolist()

                train_true_vals = np.concatenate(train_true_vals, axis=0)
                # print(train_predictions_top)
                print(train_true_vals)
                tqdm.write(f'\nEpoch {epoch}')
                loss_train_avg = loss_train_total/len(self.dataloader_train)
                tqdm.write(f'Training loss: {loss_train_avg}')
                ## calculate train metrics
                # train_accuracy = accuracy_score_func(train_predictions, train_true_vals)
                train_metrics = self.compute_metrics(self.inverse_labels, (train_predictions, train_predictions_top, train_true_vals))
                tqdm.write(f'Train Acc: {train_metrics["acc"]}')
                tqdm.write(f'Train F1: {train_metrics["f1"]} ')
                tqdm.write(f'Train WUPS: {train_metrics["wups"]}')
                tqdm.write(f'Train MRR: {train_metrics["mrr"]}')



                val_loss, predictions, predictions_top, true_vals,_ = Evaluation( self.device,  
                                                                                 self.criterion, 
                                                                                 self.model, 
                                                                                 self.dataloader_validation)
                val_metrics = self.compute_metrics(self.inverse_labels, (predictions, predictions_top , true_vals))
                ## calculate train metrics
                # val_f1 = accuracy_score_func(predictions, true_vals)
                tqdm.write(f'Validation loss: {val_loss}')
                tqdm.write(f'Val Acc: {val_metrics["acc"]}')
                tqdm.write(f'Val F1: {val_metrics["f1"]} ')
                tqdm.write(f'Val WUPS: {val_metrics["wups"]}')
                tqdm.write(f'Val MRR: {val_metrics["mrr"]}')
                val_f1 = val_metrics["wups"]

                if val_f1 >= max_auc_score:
                    tqdm.write('\nSaving best model')
                    torch.save(self.model.state_dict(), f'/content/drive/MyDrive/Datasets/models/vqa_finetuned_epoch_{epoch}_w5000.model')
                    max_auc_score = val_metrics["wups"]

                self.train_losses.append(loss_train_avg)
                self.val_losses.append(val_loss)
                self.train_f1s.append(train_metrics["wups"])
                self.val_f1s.append(val_metrics["wups"])

                self.train_metrics_dict[epoch] = train_metrics
                self.val_metrics_dict[epoch] = val_metrics

                if min_val_loss < 0:
                    min_val_loss = val_loss
                else:
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= self.early_stopping_epoch:
                            early_stop = True
                            break
                        else:
                            continue


            if early_stop:
                print("Early Stopping activated at epoch -", epoch )
                print("Use the checkpoint at epoch - ", epoch - self.early_stopping_epoch)

            # train_history.close()
            logger.log_message("info", "Training completed!")
            return self.train_losses, self.val_losses, self.train_metrics_dict, self.val_metrics_dict
        
        except Exception as e:
            print(MyException("Error training model", e))


