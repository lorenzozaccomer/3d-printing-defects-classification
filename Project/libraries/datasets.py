###
# 
#  datasets.py
#
###

import os
import logging

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from libraries.dirs import ReturnDirectories

dataset_transformation = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


img_transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 ])

# FUNCTIONS

def ShuffleDatasets(dataset, subs):
    "This function takes the datasets and shuffle them"

    """Not set num_workers because this settings can create compiling error,
    so leave the default value"""
    return {x: DataLoader(dataset[x], batch_size=4, shuffle=True) for x in subs}


def ImagesDatasetFromFolders(path, subs):
    '''
    Will create 2 dataset, one from train folder
    and one from valid folder
    '''
    return {x: datasets.ImageFolder(os.path.join(path, x), dataset_transformation[x]) for x in subs}


def ImagesDatasetSize(dataset, subs):
    '''
    Return the length of the image datasets
    '''
    return {x: len(dataset[x]) for x in subs}


def CreateAndShuffleDatasetFromPath(IMAGE_PATH):
    """
    This function create the image dataset and then shuffle them
    from the default path
    
    Return: the datasets, labels, dataset_sizes
    """
    
    SUB_DIRECTORIES = ReturnDirectories(IMAGE_PATH)

    image_datasets = ImagesDatasetFromFolders(IMAGE_PATH, SUB_DIRECTORIES)
    mixed_datasets = ShuffleDatasets(image_datasets, SUB_DIRECTORIES)

    dataset_sizes = ImagesDatasetSize(image_datasets, SUB_DIRECTORIES)
    mixed_datasets_sizes = ImagesDatasetSize(mixed_datasets, SUB_DIRECTORIES)

    # Extract the labels from one dataset (are equal between them)
    labels = image_datasets['train'].classes

    logging.debug("dataset sizes: " + str(dataset_sizes))
    logging.debug("mixed_datasets sizes: " + str(mixed_datasets_sizes))

    return image_datasets, mixed_datasets, labels, dataset_sizes