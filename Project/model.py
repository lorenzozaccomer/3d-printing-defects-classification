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



def model_generation(IMAGE_PATH, MODEL_PATH, iteration = 0, visualize_prediction = 1, EPOCH_NUMBER = 1, LEARNING_RATE = 0.05):
    """
    This function will generate the model for image classification
    that you desire, by default the iteration flag is equal to 0,
    so in this case it loads the model from the MODEL_PATH, 
    otherwise with iteration = 1 you create a new model and you can visualize
    its prediction
    """

    SUB_DIRS = CheckDirectories(IMAGE_PATH)

    logging.debug("iteration: " + str(iteration))
    logging.debug("visualize_prediction: " + str(visualize_prediction))
    logging.debug("EPOCH_NUMBER: " + str(EPOCH_NUMBER))
    logging.debug("LEARNING_RATE: " + str(LEARNING_RATE))

    logging.info("IMAGE_PATH: " + IMAGE_PATH)
    logging.info("SUB_DIRS: " + str(SUB_DIRS))

    image_datasets = ImagesDatasetFromFolders(IMAGE_PATH, SUB_DIRS)

    mixed_datasets = ShuffleDatasets(image_datasets, SUB_DIRS)

    dataset_sizes = ImagesDatasetSize(image_datasets, SUB_DIRS)

    # Extract the labels from one dataset (are equal between them)
    labels = image_datasets['train'].classes

    logging.debug("dataset_sizes: " + str(dataset_sizes))
    logging.debug("len(mixed_datasets): " + str(len(mixed_datasets)))
    for label in ['train', 'valid']:
        logging.debug("len(mixed_datasets): " + str(len(mixed_datasets[label])))

    # generate a new model and save it on the path
    # that you choose
    if iteration == 1:

        logging.info("-----------   NEW ITERATION  -----------")
        print("loading model generation ..")
        logging.info("loading model generation ..")

        pretrained_model = models.resnet50(pretrained=True)
        
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

    # if iteration = 0 and visualize_prediction is = 1
    # the script load the model from the path that you choose and
    # visualize the prediction
    elif iteration == 0 and visualize_prediction == 1:
        print("loading of default model from path ..")
        logging.info("-----------   LOAD MODEL  -----------")
        logging.info("skipped a new model generation ..")
        logging.info("loading of default model from path ..")
        
        loaded_model = torch.load(MODEL_PATH)

        visualize_generated_model(mixed_datasets, labels, loaded_model)
        print("closing ..")