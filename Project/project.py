
# Libraries
import cv2
import torch

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import models
import matplotlib.pyplot as plt

# importing Functions from files
from libs.dirs import *
from libs.datasets import *
from libs.training import *
from libs.visualizer import *

cudnn.benchmark = True
plt.ion()   # interactive mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DesiredPath = "U:\\repositories\\3d-printer-recognition\\Images"

Path, Subpaths = CheckCurrentPathAndExtractSubPaths(DesiredPath)

print(Path)
print(Subpaths)

image_datasets = ImagesDatasetFromFolders(Path, Subpaths)

dataloaders = ShuffleDatasets(image_datasets, Subpaths)

dataset_sizes = ImagesDatasetSize(image_datasets, Subpaths)
print(dataset_sizes)

# # Extract the class from one dataset (are equal between them)
class_names = image_datasets['train'].classes

#generate_batch_images(dataloaders,class_names, 'train')

model_ft = models.resnet18(pretrained=True)

for param in model_ft.parameters():
    param.requires_grad = False

num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

generated_model = train_model(dataloaders, dataset_sizes, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1)

visualize_model(dataloaders, class_names, generated_model)

plt.ioff()
plt.show()

def save_model(model, optimizer):
    checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
    torch.save(checkpoint, './alexnet_extractor_model.pth')

save_model(generated_model, optimizer_ft)
