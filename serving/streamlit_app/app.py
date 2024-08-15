import streamlit as st
import numpy as np
import pandas as pd
import torch
import json

from PIL import Image
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel
from infer import InfenceTest
from model import main

## Read file all of class
with open("dataclass/inverse_labels.json", "r") as js:
     inverse_labels = json.load(js)


#  Inititalize model
model_name = "vikenkd/vqa-llm"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Load the model and tokenizer
model = AutoModel.from_pretrained(model_name)
model = main()
model = model.to(device)

# Load tokenize
visual_feature_extractor_name = "google/vit-base-patch16-224-in21k"
textual_feature_extractor_name = "bert-base-uncased"

## Text
tokenizer = AutoTokenizer.from_pretrained(textual_feature_extractor_name)
text_encoder = AutoModel.from_pretrained(textual_feature_extractor_name)
for p in text_encoder.parameters():
    p.requires_grad = False

## Image processor
image_processor = AutoFeatureExtractor.from_pretrained(visual_feature_extractor_name)
image_encoder = AutoModel.from_pretrained(visual_feature_extractor_name)

for p in image_encoder.parameters():
    p.requires_grad = False


image_encoder = image_encoder.to(device)
text_encoder = text_encoder.to(device)

## Initialize class
infer_encoding = InfenceTest(image_encoder, 
                             text_encoder, 
                             tokenizer, 
                             image_processor,
                             device)

# Custom Website App for deploying that model

st.title("📝 Visual Question Answering Project", )

width_head = """
    <style>
        [data-testid="stApp"] {
                background: url(https://img.freepik.com/free-photo/vivid-blurred-colorful-background_58702-2655.jpg);
                background-attachment: fixed;
                background-repeat: no-repeat;
                background-size: cover;
        }
        [data-testid="StyledLinkIconContainer" span] {
            white-space : nowrap;
            background: -webkit-linear-gradient(#eee, #333);
            -webkit-background-clip: text;
            # -webkit-text-fill-color: transparent;
        }   
        .st-emotion-cache-zt5igj span {
            white-space: nowrap;
            background: linear-gradient(172deg, #c42525cf, #2130ae);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        [data-testid="element-container"]{
                text-align: -webkit-center;

        }
        [data-testid="baseButton-secondaryFormSubmit"]{
            padding: 12px 40px;
        }
    </style>
"""
st.markdown(width_head, unsafe_allow_html=True)
# Custom CSS to create a border
border_css = """
<style>
    .custom-border {
        border: 10px solid #4CAF50;  /* Green border */
        padding: 10px;
        border-radius: 5px;
    }
</style>
"""

st.markdown('<div class="custom-border">', unsafe_allow_html=True)
# st.write("This is inside a bordered box")
st1, st2 = st.columns(2)
image = None
if image == None:
    uploaded_file = st.file_uploader("Upload an article", type=("png", "jpg"))
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Open the image using PIL
        image = Image.open(uploaded_file)
        
question: str = ""
submitted = None
if image != None:
        with st.form("my_form"):
            col1, col2 = st.columns(2)
            with col1:
                question = st.text_area("Enter text:", placeholder="What is your question?")
            with col2:
                st.image(image, caption="Uploaded Image.", use_column_width=True)
            if not image or  not question:
                if not image:
                     st.info("Please upload your image.")
                elif not question:
                     st.info("Please add your question.")
            
            submitted = st.form_submit_button("Submit")     

st.markdown('</div>', unsafe_allow_html=True)

inverse_labels = None

if submitted:
    encoding_status = infer_encoding.encoding(question = question, image = image)
    answer = infer_encoding.infer(model= model, 
                         inputs_require= encoding_status, 
                         top_k= 10)
    st.write("### Answer")
    st.write(inverse_labels[answer["answer"]])
    st.write(" - With answer's probability is ")
    st.write(answer["probs"])
