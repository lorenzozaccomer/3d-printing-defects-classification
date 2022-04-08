
# Libraries
import cv2
import torch, torchvision

from torchvision import datasets

# Standard Library
import os, sys

# importing Functions from files
from libs.dirs import *
from libs.datasets import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
# extract the train-valid images folders
Path = 'U:\\repositories\\3d-printer-recognition\\Images'
Subpaths = ['train','valid']

image_datasets = ImagesDatasetFromFolders(Path, Subpaths)
#print(image_datasets)

dataloaders = ShuffleDatasets(image_datasets, Subpaths)
print(dataloaders)

dataset_sizes = ImagesDatasetSize(image_datasets, Subpaths)
print(dataset_sizes)

# Extract the class from one dataset (are equal between them)
class_names = image_datasets['train'].classes
print(class_names)

# Get 1 batch images
inputs, classes = next(iter(dataloaders['train']))

# Get a batch of training data
out = torchvision.utils.make_grid(inputs)

# Print 4 random images
imshow(out, title=[class_names[x] for x in classes])

print("ok")

