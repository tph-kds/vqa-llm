
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

# def infer():
#     pass

class InfenceTest:

    def __init__(self, 
                 image_encoder,
                 text_encoder,
                 tokenizer,
                 image_processor,
                #  batch_size, 
                 device):
        
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        # self.batch_size = batch_size

        self.device = device


    def encoding(self, question, image):

        # image_file = image
        question = question
        # full_path = self.type_data + "/images/" + image_file
        # source_image = os.path.join(os.getcwd(), full_path)
        # image = Image.open(image_file).convert("RGB")

        image_inputs = self.image_processor(image, return_tensors="pt")
        image_inputs = {k:v.to(self.device) for k,v in image_inputs.items()}
        image_outputs = self.image_encoder(**image_inputs)
        image_embedding = image_outputs.pooler_output
        image_embedding = image_embedding.view(-1)
        image_embedding = image_embedding.detach()

        text_inputs = self.tokenizer(question, return_tensors="pt")
        text_inputs = {k:v.to(self.device) for k,v in text_inputs.items()}
        text_outputs = self.text_encoder(**text_inputs)
        text_embedding = text_outputs.pooler_output # You can experiment with this or raw CLS embedding below
        text_embedding = text_embedding.view(-1)
        text_embedding = text_embedding.detach()

        encoding={}
        encoding["image_emb"] = image_embedding
        encoding["question_emb"] = text_embedding

        return encoding
    
    def infer(self, model, inputs_require, top_k: int = 10):
        
        inputs = {
            'image_emb':  inputs_require["image_emb"].unsqueeze(0),
            'question_emb': inputs_require["question_emb"].unsqueeze(0)
        }

        with torch.no_grad():
            outputs = model(**inputs)

        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)
        # Get top 10 probabilities and their indices for each example in the batch

        top_probabilities, top_indices = torch.topk(probabilities, k=top_k, dim=1)
        # print(top_indices.shape)
        top_indices = top_indices.detach().cpu().numpy()

        probs   = torch.max(outputs.softmax(dim=1), dim=-1)[0].detach().cpu().numpy()
        outputs = outputs.argmax(-1)
        logits = outputs.detach().cpu().numpy()

        return {
                "answer": logits, 
                "probs": probs, 
                "topk" : top_indices, 
                "topk_probs" : top_probabilities
        }



