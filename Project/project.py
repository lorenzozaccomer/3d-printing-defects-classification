
# Libraries
import torch
import logging

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

logging.basicConfig(filename='log_image_classification.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_PATH = "L:\\repositories\\3d-printer-recognition\\Images"
MODEL_PATH = './generated_model.pth'

Path, Subpaths = CheckCurrentPathAndExtractSubPaths(IMAGE_PATH)

logging.info("-----------   NEW EXECUTION  -----------")
logging.info("Path: " + Path)
logging.info("Subpaths: " + str(Subpaths))

iteration = 0 # 1 to skip model generation

image_datasets = ImagesDatasetFromFolders(Path, Subpaths)

mixed_datasets = ShuffleDatasets(image_datasets, Subpaths)
logging.debug("len(mixed_datasets['train']): " + str(len(mixed_datasets['train'])))
logging.debug("len(mixed_datasets['valid']): " + str(len(mixed_datasets['valid'])))

dataset_sizes = ImagesDatasetSize(image_datasets, Subpaths)
logging.debug("dataset_sizes: " + str(dataset_sizes))

# Extract the class from one dataset (are equal between them)
labels = image_datasets['train'].classes

if iteration == 0: # I want to generate a model

    print("loading model generation ..")
    logging.info("loading model generation ..")

    model_ft = models.resnet50(pretrained=True)

    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(labels)).
    model_ft.fc = nn.Linear(num_ftrs, len(labels))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    starting_time = time.time()
    generated_model = train_model(mixed_datasets, dataset_sizes, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1)
    logging.info('Training time: {:10f} minutes'.format((time.time()-starting_time)/60))
    
    logging.info("saving the model ..")
    torch.save(generated_model, MODEL_PATH)
    logging.info("model saved!")
    
    logging.info("visualize generated model ..")
    visualize_generated_model(mixed_datasets, labels, generated_model)
    print("closing ..")

else:
    logging.info("skipped a new model generation ..")
    logging.info("default model loading ..")
    
    loaded_model = torch.load(MODEL_PATH)

    visualize_generated_model(mixed_datasets, labels, loaded_model)
