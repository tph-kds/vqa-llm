import glob
import os
import shutil
import sys
import time
import json
import random
from typing import Dict, List
import requests
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.vqa_llm.logger import logger
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.utils.file import File

class PrepareImageData:
    def __init__(self,
                 current_folder:str, 
                 source_folder:str, 
                 destination_folder:str, 
                 images_folder:str):
        
        self.current_folder = current_folder
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.source_dir = os.path.join(current_folder, source_folder)
        self.destination_dir = os.path.join(current_folder, destination_folder)

        self.images_folder = images_folder

    def check_folder_exist(self):

        try:
            # Ensure the destination directory exists
            if not os.path.exists(self.destination_dir):
                os.makedirs(self.destination_dir)
                logger.log_message("info", f"Directory at {self.destination_dir} created successfully!")

        except Exception as e:
            print(MyException("Error checking folder", e))

    def count_image(self, ):
        try:
            for name in os.listdir(self.source_dir):
                sour_name = os.path.join(self.source_dir, name)
                print(sour_name)
                print(len(sour_name))
                # Define the image extensions you want to count
                image_extensions = ['*.jpg','*.png']

                # Count the total number of images
                total_images = 0

                for ext in image_extensions:
                    # Use glob to find all files with the given extension
                    total_images += len(glob.glob(os.path.join(sour_name, ext)))
                logger.log_message("info", f"Total number of images: {total_images}")

        except Exception as e:
            print(MyException("Error Counting Image", e))


    def image_folder_dataset(self, sample_train: int = 10000):
        try:
            for name in os.listdir(self.source_dir):
                sour_name = os.path.join(self.source_dir, name)
                des_name = os.path.join(self.estination_dir, name)

                # check folder for destination folder
                # Ensure the destination directory exists
                images_folder_des = os.path.join(des_name, self.images_folder)
                if not os.path.exists(images_folder_des):
                    os.makedirs(images_folder_des)
                    logger.log_message("info", f"Directory at {images_folder_des} created successfully!")

                if name == "train":
                    sample_count = sample_train
                else: 
                    sample_count = sample_train / 2
                count = 0

                for img in os.listdir(sour_name):
                    if img != "questions" and img != "answers":

                        source_img_folder = os.path.join(sour_name, img)
                        destination_img_folder = images_folder_des

                        for img_file in os.listdir(source_img_folder):
                            # print(str(int(img_file.split(".")[0].split("_")[-1])))
                            img_idx_str = str(int(img_file.split(".")[0].split("_")[-1]))
                            end_img = str(img_file.split(".")[-1])
                            img_file_new = img_idx_str + "." + end_img

                            source_img_file = os.path.join(source_img_folder, img_file)
                            destination_img_file = os.path.join(destination_img_folder, img_file_new)


                            # Move the file
                            shutil.copy2(source_img_file, destination_img_file)
                            if count == sample_count:
                                break
                            count += 1

                logger.log_message("info", f"Move {count} images from {sour_name} to {des_name} successfully!")


        except Exception as e:
            print(MyException("Error splitting image processing", e))


if __name__ == "__main__":
    
    source_folder = 'datasets'
    destination_folder = 'split_datasets'
    current_folder = os.getcwd()
    
    images_folder = "images"

    prepare_image_data = PrepareImageData(current_folder, source_folder, destination_folder)
    prepare_image_data.check_folder_exist()
    prepare_image_data.image_folder_dataset()
    
