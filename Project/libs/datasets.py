###
# 
#  datasets.py
#
###

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(800),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(600),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# FUNCTIONS

def ShuffleDatasets(dataset, subs):
    "This function takes the datasets and shuffle them"

    """Not set num_workers because this settings can create compiling error,
    so leave the default value"""
    return {x: DataLoader(dataset[x], batch_size=4, shuffle=True, num_workers=4) for x in subs}


def ImagesDatasetFromFolders(path, subs):
    '''
    Will create 2 dataset, one from train folder
    and one from valid folder
    '''
    return {x: datasets.ImageFolder(os.path.join(path, x), data_transforms[x]) for x in subs}


def ImagesDatasetSize(dataset, subs):
    '''
    Return the length of the image datasets
    '''
    return {x: len(dataset[x]) for x in subs}