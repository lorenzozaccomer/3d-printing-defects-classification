
# Libraries
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

DesiredPath = "L:\\repositories\\3d-printer-recognition\\Images"

Path, Subpaths = CheckCurrentPathAndExtractSubPaths(DesiredPath)

print(Path)
print(Subpaths)

image_datasets = ImagesDatasetFromFolders(Path, Subpaths)

mixed_datasets = ShuffleDatasets(image_datasets, Subpaths)
print(len(mixed_datasets['train']))

dataset_sizes = ImagesDatasetSize(image_datasets, Subpaths)
print(dataset_sizes)

# # Extract the class from one dataset (are equal between them)
class_names = image_datasets['train'].classes

#generate_batch_images(mixed_datasets['train'],class_names)



PATH = './model_ft.pth'
iteration = 1 #skip model generation
model_visualization = 0 #skip visualize_model



if iteration == 0: # I want to generate a model

    print("model generation ..")

    model_ft = models.resnet50(pretrained=True)

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

    starting_time = time.time()
    generated_model = train_model(mixed_datasets, dataset_sizes, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)
    print('Training time: {:10f} minutes'.format((time.time()-starting_time)/60))
    
    print("saving the model ..")
    torch.save(generated_model, PATH)
    print("saved!")
    
    if(model_visualization == 0): # only model geneartion
        exit()

if(model_visualization == 1):
    visualize_model(mixed_datasets, class_names, generated_model)
    plt.ioff()
    plt.show()

if iteration == 1:
    print("model loading ..")
    
    loaded_model = torch.load(PATH)
    
    images, labels = next(iter(mixed_datasets['valid']))
    
    # print images
    imshow(torchvision.utils.make_grid(images), title=[class_names[x] for x in labels])

    outputs = loaded_model(images)
    labels = torch.tensor(labels)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ',' '.join('%s' % class_names[predicted[j]] for j in range(images.size()[0])))

    plt.ioff()
    plt.show()
