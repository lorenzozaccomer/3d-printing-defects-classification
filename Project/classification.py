###
# 
#  classification.py
#
###

from PIL import Image

import torch
import logging
import argparse
import matplotlib.pyplot as plt

# importing Functions from files
from libraries.visualizer import imshow
from libraries.datasets import img_transformation
from libraries.errors import *

from model import *

plt.ion()   # interactive mode

# Remove unwanted log information
logging.getLogger('PIL').setLevel(logging.WARNING)

parser = argparse.ArgumentParser(description='3d Printer Image Classification')

# Command line arguments
parser.add_argument('--dataset_images_path', type = str, default = './Images/3d/', help = 'Is the path of your image datasets')
parser.add_argument('--epochs', type = int, default = 25, help = 'Epoch number')
parser.add_argument('--test_image_path', type = str, default = './Images\\classification-images\\7.jpg', help = 'Is the path for your test image classification')
parser.add_argument('--iteration', type = int, default = 0, help = 'Iteration')
parser.add_argument('--learning_rate', type = float, default = 0.05, help = 'Learning Rate')
parser.add_argument('--model_path', type = str, default = './generated_model.pth', help = 'Path of your model')
parser.add_argument('--visualize_prediction', type = int, default = 0, help = 'Visualize Prediction')

option = parser.parse_args()
print(option)

labels = ['NoDefects', 'YesDefects']

CheckParametersErrors(option.epochs, option.learning_rate, option.iteration, 
    option.visualize_prediction, option.test_image_path, 
    option.model_path, option.dataset_images_path)

if option.iteration == 1:
    print('execution of model generation function')
    ModelGeneration(option.dataset_images_path, option.model_path, option.epochs, option.learning_rate)
elif option.iteration == 0 and option.visualize_prediction == 1:
    print('skip model generation, it loads the model from path and visualize the results')
    LoadModelVisualization(option.dataset_images_path, option.model_path)
    exit()
else:
    print("loading image classification ..")


logging.info("-----------   NEW IMAGE CLASSIFICATION  -----------")
logging.info("loading image classification ..")

# Load model
evaluation_model = torch.load(option.model_path)
evaluation_model.eval()

#Load image
img = Image.open(option.test_image_path)
image = img_transformation(img)

# Carry out inference
tensor_image = image.unsqueeze(0)
out = evaluation_model(tensor_image)

_, indices = torch.max(out,1)

percentage = torch.softmax(out, dim=1)[0] * 100

prediction_text = "Prediction: " + labels[indices.item()] + " " + str(percentage[indices].item())

imshow(image, prediction_text)

plt.ioff()
plt.show()

logging.info(prediction_text)
logging.info("-----------   END IMAGE CLASSIFICATION  -----------")