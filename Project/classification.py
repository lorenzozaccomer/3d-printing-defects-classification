

from PIL import Image
from numpy import indices

import torch
import logging
import argparse
import os
import matplotlib.pyplot as plt

# importing Functions from files
from libraries.visualizer import imshow
from libraries.datasets import img_transformation
from model import model_generation

plt.ion()   # interactive mode

parser = argparse.ArgumentParser(description='3d Printer Image Classification')

# Command line arguments
parser.add_argument('--dataset_images_path', type = str, default = './Images', help = 'Is the path of your image datasets')
parser.add_argument('--epochs', type = int, default = 1, help = 'Epoch number')
parser.add_argument('--image_path_file', type = str, default = './Project\\test\\7.jpg', help = 'Is the path for your test image classification')
parser.add_argument('--iteration', type = int, default = 0, help = 'Iteration')
parser.add_argument('--learning_rate', type = float, default = 0.05, help = 'Learning Rate')
parser.add_argument('--model_path', type = str, default = './generated_model.pth', help = 'Path of your model')
parser.add_argument('--visualize_prediction', type = int, default = 0, help = 'Visualize Prediction')

opt = parser.parse_args()
print(opt)

labels = ['NoDefects', 'YesDefects']

if opt.iteration == 1:
    print('execution of model generation function')
    model_generation(opt.dataset_images_path, opt.model_path, opt.iteration, opt.visualize_prediction, opt.epochs, opt.learning_rate)
else:
    print('skip model generation, it loads the model from path and visualize the results')

# check if the image path exists
if not(os.path.exists(opt.image_path_file)):
    print("path file error! this image file doesn't exist")
    exit()

logging.info("-----------   NEW IMAGE CLASSIFICATION  -----------")
print("loading image classification ..")
logging.info("loading image classification ..")

# Load model
evaluation_model = torch.load(opt.model_path)
evaluation_model.eval()

#Load image
img = Image.open(opt.image_path_file)
image = img_transformation(img)

# Carry out inference
tensor_image = image.unsqueeze(0)
out = evaluation_model(tensor_image)

_, indices = torch.max(out,1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

prediction_text = "Prediction: " + labels[indices.item()] + " " + str(percentage[indices].item())

imshow(image, prediction_text)

plt.ioff()
plt.show()