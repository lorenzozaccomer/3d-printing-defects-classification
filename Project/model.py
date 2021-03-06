###
# 
#  model.py
#
###

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
from libraries.dirs import *
from libraries.datasets import *
from libraries.training import *
from libraries.visualizer import *
from libraries.constants import *

cudnn.benchmark = True
plt.ion()   # interactive mode

logging.basicConfig(filename='log_generation_model.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True



def ModelGeneration(IMAGE_PATH, MODEL_PATH, EPOCH_NUMBER = 1, LEARNING_RATE = 0.05):
    """
    This function will generate the model for image classification
    that you desire, by default the iteration flag is equal to 0,
    so in this case it loads the model from the MODEL_PATH, 
    otherwise with iteration = 1 you create a new model and you can visualize
    its prediction
    """

    _, mixed_datasets, labels, dataset_sizes = CreateAndShuffleDatasetFromPath(IMAGE_PATH)

    logging.info("-----------   NEW ITERATION  -----------")
    print("loading model generation ..")
    logging.info("loading model generation ..")

    pretrained_model = models.resnet50(pretrained=True)
    
    # To reduce training time
    for param in pretrained_model.parameters():
        param.requires_grad = False

    num_ftrs = pretrained_model.fc.in_features

    pretrained_model.fc = nn.Linear(num_ftrs, len(labels))

    pretrained_model = pretrained_model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(pretrained_model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    starting_time = time.time()
    generated_model = train_model(mixed_datasets, dataset_sizes, pretrained_model, criterion, optimizer_ft, exp_lr_scheduler, EPOCH_NUMBER)
    logging.info('Training time: {:10f} minutes'.format((time.time()-starting_time)/60))
    
    logging.info("saving the model ..")
    torch.save(generated_model, MODEL_PATH)
    logging.info("model saved!")
    
    logging.info("visualize generated model ..")
    visualize_generated_model(mixed_datasets, labels, generated_model)
    print("closing ..")


def LoadModelVisualization(IMAGE_PATH, MODEL_PATH):
    """
    This function load the model previously created from the path and
    allow to visualize the prediction
    """

    print("loading of default model from path ..")
    logging.info("-----------   LOAD MODEL  -----------")
    logging.info("skipped a new model generation ..")
    logging.info("loading of default model from path ..")

    _, mixed_datasets, labels, _ = CreateAndShuffleDatasetFromPath(IMAGE_PATH)
    
    loaded_model = torch.load(MODEL_PATH)

    visualize_generated_model(mixed_datasets, labels, loaded_model)
    print("closing ..")