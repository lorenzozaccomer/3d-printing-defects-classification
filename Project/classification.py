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
from libraries.constants import *

plt.ion()   # interactive mode

# Remove unwanted log information
logging.getLogger('PIL').setLevel(logging.WARNING)

parser = argparse.ArgumentParser(description='3d Printer Image Classification')

# Command line arguments
parser.add_argument('--dataset_path', type = str, default = './Images/3d/', help = 'Is the path of your image datasets')
parser.add_argument('--epochs', type = int, default = 25, help = 'Epoch number')
parser.add_argument('--test_image_path', type = str, default = './Images/classification-images/7.jpg', help = 'Is the path for your test image classification')
parser.add_argument('--iteration', type = int, default = 0, help = 'Iteration')
parser.add_argument('--learning_rate', type = float, default = 0.05, help = 'Learning Rate')
parser.add_argument('--model_path', type = str, default = './generated_model.pth', help = 'Path of your model')
parser.add_argument('--visualize_prediction', type = int, default = 0, help = 'Visualize Prediction')

options = parser.parse_args()
for opt in options.__dict__.items():
    print(str(opt))

CheckParametersErrors(
    options.epochs, options.learning_rate, options.iteration, 
    options.visualize_prediction, options.test_image_path, 
    options.model_path, options.dataset_path   )


# Load model
evaluation_model = torch.load(options.model_path)
evaluation_model.eval()

#Load image
img = Image.open(options.test_image_path)
transformed_image = img_transformation(img)

# Carry out inference
tensor_image = transformed_image.unsqueeze(0)
out = evaluation_model(tensor_image)

_, indices = torch.max(out,1)

percentage = torch.softmax(out, dim=1)[0]*100

percentage = "{:.5f}".format(percentage[indices].item())
label = labels[indices.item()]

prediction_text = "Prediction: " + label + " " + str(percentage)
print("Prediction: " + label + " " + str(percentage))

imshow(transformed_image, prediction_text)

plt.ioff()
plt.show()

logging.info(prediction_text)
logging.info("-----------   END IMAGE CLASSIFICATION  -----------")
print("End of Image Classification")