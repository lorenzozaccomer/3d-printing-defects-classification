

from PIL import Image
from numpy import indices

import torch
import logging
import argparse
import os
import matplotlib.pyplot as plt

# importing Functions from files
from libs.visualizer import imshow
from libs.datasets import img_transformation
from project import model_generation

parser = argparse.ArgumentParser(description='3d Printer Image Classification')

# Command line arguments
parser.add_argument('--learning_rate', type = float, default = 0.05, help = 'Learning Rate')
parser.add_argument('--epochs', type = int, default = 1, help = 'Epochs')
parser.add_argument('--model_path', type = str, default = './generated_model.pth', help = 'Path of your model')
parser.add_argument('--iteration', type = int, default = 0, help = 'Iteration')
parser.add_argument('--visualize_prediction', type = int, default = 0, help = 'Visualize Prediction')

opt = parser.parse_args()
print(opt)

plt.ion()   # interactive mode

MODEL_PATH = './generated_model.pth'
IMAGE_DATASET_PATH = './Images'
IMAGE_PATH_FILE = "./Project\\test\\7.jpg"
classes = ['NoDefects', 'YesDefects']

iteration = 0
visualize_prediction = 0

model_generation(IMAGE_DATASET_PATH, MODEL_PATH, iteration, visualize_prediction)

if iteration == 1:
    print('execution of model generation function')
else:
    print('skip model generation, it loads the model from path and visualize the results')

# check if the image path exists
if not(os.path.exists(IMAGE_PATH_FILE)):
    print("path file error! this image file doesn't exist")
    exit()

logging.info("-----------   NEW IMAGE CLASSIFICATION  -----------")
print("loading image classification ..")
logging.info("loading image classification ..")

# Load model
evaluation_model = torch.load(MODEL_PATH)
evaluation_model.eval()

#Load image
img = Image.open(IMAGE_PATH_FILE)
image = img_transformation(img)

# Carry out inference
tensor_image = image.unsqueeze(0)
out = evaluation_model(tensor_image)

_, indices = torch.max(out,1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

prediction_text = "Prediction: " + classes[indices.item()] + " " + str(percentage[indices].item())

imshow(image, prediction_text)

plt.ioff()
plt.show()