

from PIL import Image
from numpy import indices

import torch
import logging
import matplotlib.pyplot as plt

# importing Functions from files
from libs.visualizer import imshow
from libs.datasets import img_transformation
from project import model_generation

plt.ion()   # interactive mode

MODEL_PATH = 'L:\\Università\\repositories\\3d-printer-recognition\\generated_model.pth'
IMAGE_DATASET_PATH = 'L:\\Università\\repositories\\3d-printer-recognition\\Images'
IMAGE_PATH = "L:\\Università\\repositories\\3d-printer-recognition\\Project\\test\\7.jpg"
classes = ['NoDefects', 'YesDefects']

iteration = 0
visualize_prediction = 0

model_generation(IMAGE_DATASET_PATH, MODEL_PATH, iteration, visualize_prediction)

if iteration == 1:
    print('execution of model generation function')
else:
    print('skip model generation, it loads the model from path and visualize the results')

logging.info("-----------   NEW IMAGE CLASSIFICATION  -----------")
print("loading image classification ..")
logging.info("loading image classification ..")

# Load model
evaluation_model = torch.load(MODEL_PATH)
evaluation_model.eval()

#Load image
img = Image.open(IMAGE_PATH)
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