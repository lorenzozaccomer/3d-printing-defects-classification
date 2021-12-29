
# Libraries
import cv2
import torch, torchvision, numpy, matplotlib

from torchvision import datasets

# Standard Library
import os, sys

# importing Functions from files
from libs.dirs import *
from libs.datasets import *
 
# extract the train valid folders
Path = 'D:\\repositories\\3d-printer-recognition\\Images'
Subpaths = ['train','valid']

image_datasets = ImagesDatasetFromFolders(Path, Subpaths)
print(image_datasets)

dataset_sizes = ImagesDatasetSize(image_datasets, Subpaths)
print(dataset_sizes)

