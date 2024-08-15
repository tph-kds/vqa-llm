import torch
from transformers import PreTrainedModel, PretrainedConfig
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn
import math
from torch.nn import functional as F

# import torchtext
from torch.utils.data import dataset

embed_dim : int  = 512
num_layers: int = 2
dropout: float = 0.3
input_layer: int = 128
output_layer: int = 256
n_dense: int = 6
batchnorm_layer: int = 256
output_classification: int = 3454
w_s: float = 0.5
learning_rate: float = 0.001
batch_size: int = 32
num_epochs: int = 20
expansion_factor: int = 4
n_heads = 8
hidden_dim: int = 128

train_steps: int =20000
learning_rate: float = 5e-5
weight_decay: float = 1e-5
eps : float = 1e-8
warm_steps: int = train_steps * 0.1


class CSCA(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(CSCA, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoder(TransformerEncoderLayer(d_model=d_model, nhead=nhead), num_layers=1)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x = torch.cat((image_features, text_features), dim=1)
        for layer in self.layers:
            x = layer(x)
        return x

csca = CSCA(d_model=n_dense*input_layer, nhead=8, num_layers=2)


class FusionModel(nn.Module):
  def __init__(self,
               embed_dim,
               dropout,
               n_dense,
               batchnorm_layer,
               input_layer,
               output_layer):

        super(FusionModel, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, output_layer)
        self.bn1 = nn.BatchNorm1d(batchnorm_layer)

  def forward(self, x1):
      # print(x1.shape)
      part_2 = x1.shape[1] - self.embed_dim
      x = torch.mul(x1[:, :self.embed_dim], x1[:, part_2:])
      # print(x.shape)
      x = self.fc1(x)
      x = self.bn1(x)
      x = self.dropout(x)

      return x
  

class AnswerPredictor(nn.Module):
  def __init__(self,
               input_dim,
               hidden_dim ,
               output_dim):
    super(AnswerPredictor, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.relu_f = nn.ReLU()
    self.fc2 = nn.Linear(hidden_dim, output_dim)


  def forward(self, x):
    x = self.fc1(x)
    x = self.relu_f(x)
    x = self.fc2(x)

    return x


class VqaMidFusionNetwork(nn.Module):

    def __init__(self,
                 attentionblock,
                 fusion_features,
                 w_s,
                 answer_predictor):

        super(VqaMidFusionNetwork, self).__init__()
        # self.dropout = nn.Dropout(dropout)
        # self.fc1 = nn.Linear(n_dense * input_layer, output_layer)
        # self.bn1 = nn.BatchNorm1d(batchnorm_layer)
        # self.classifier = nn.Linear(output_layer, output_classification)

        # FusionModel
        self.fusion_features = fusion_features

        W = torch.Tensor(n_dense * input_layer, n_dense * input_layer)
        self.W = nn.Parameter(W)
        # initialize weight matrices
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(w_s))

        self.relu_f = nn.ReLU()

        # self.attentionblock = nn.ModuleList(
        #     [
        #         SelfAttentionBlock(embed_dim, expansion_factor, n_heads)
        #         for i in range(num_layers)
        #     ]
        # )

        self.attentionblock = attentionblock

        self.answer_predictor = answer_predictor


    def forward(self, image_emb, question_emb):

        x1 = image_emb
        Xv = torch.nn.functional.normalize(x1, p=2, dim=1)
        # print(Xv.shape)

        x2 = question_emb
        Xt = torch.nn.functional.normalize(x2, p=2, dim=1)
        # print(Xt.shape)

        Xvt = Xv * Xt
        # Xvt =  torch.cat((Xv, Xt), dim=1)
        # print(Xvt.shape)

        # for layer in self.layers:
        #   Xvt = layer(Xvt, Xvt, Xvt)
        # Self-Attention Block
        Xvt = self.attentionblock(Xvt)

        Xvt = self.relu_f(torch.mm(Xvt, self.W.t()))

        # Xvt = self.fc1(Xvt)
        # Xvt = self.bn1(Xvt)
        # Xvt = self.dropout(Xvt)
        Xvt = self.fusion_features(Xvt)

        Xvt = self.answer_predictor(Xvt)

        # Xvt = self.classifier(Xvt)

        return Xvt
    
def main():
    torch.cuda.empty_cache()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    fusion_features = FusionModel(embed_dim,
                                dropout,
                                n_dense,
                                batchnorm_layer,
                                input_layer,
                                output_layer)
    # Instantiate answer predictor
    answer_predictor = AnswerPredictor(output_layer,
                                    hidden_dim,
                                    output_classification)
    # Instantiate model
    # model = VqaMidFusionNetwork(attentionblock,
    model = VqaMidFusionNetwork(csca,
                                fusion_features,
                                w_s,
                                answer_predictor)
    model.to(device)
    model.load_state_dict(torch.load('models/model.model'))
    model.to(device)
    return model
