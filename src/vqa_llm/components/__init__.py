from src.vqa_llm.components.data_preparation.text_data import PrepareTextData
from src.vqa_llm.components.data_preparation.image_data import PrepareImageData
from src.vqa_llm.components.data_transformation import DataTransformation
from src.vqa_llm.components.loader_models import LoadModel

dataTransformation = DataTransformation(source_folder="src/vqa_llm/data", dotfile=".jpg")
