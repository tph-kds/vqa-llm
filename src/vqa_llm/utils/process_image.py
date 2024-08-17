from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.logger import logger
from torchvision import transforms
from PIL import Image

class ImageHandling:
    def __init__(self, 
                 image_path:str):
        self.image_path = image_path
        
    # Function to preprocess the image
    def preprocess_image(self):
        try:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            image = Image.open(self.image_path)
            logger.log_message("info", "Preprocessing image... Done!")
            return transform(image).unsqueeze(0)
        

        except Exception as e:
            print(MyException("Error preprocessing image", e))
