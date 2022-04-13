
# Libraries
import cv2
import torch, torchvision

import torch.nn as nn
import torch.optim  as optim

from torchvision import models

# importing Functions from files
from libs.dirs import *
from libs.datasets import *
from libs.training import *
from libs.visualizer import *

DesiredPath = "U:\\repositories\\3d-printer-recognition\\Images"


Path, Subpaths = CheckCurrentPathAndExtractSubPaths(DesiredPath)

print(Path)
print(Subpaths)

image_datasets = ImagesDatasetFromFolders(Path, Subpaths)
#print(image_datasets)

dataloaders = ShuffleDatasets(image_datasets, Subpaths)
#print(dataloaders)

dataset_sizes = ImagesDatasetSize(image_datasets, Subpaths)
print(dataset_sizes)

# Extract the class from one dataset (are equal between them)
class_names = image_datasets['train'].classes
#print(class_names)

#generate_batch_images(dataloaders,class_names)

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

generated_model = train_model(dataloaders, dataset_sizes, Subpaths, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)

visualize_model(dataloaders, class_names, generated_model)

plt.ioff()
plt.show()