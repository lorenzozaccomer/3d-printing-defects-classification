
import os, torch 
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

data_transforms = {
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

# FUNCTIONS

def ShuffleDatasets(dataset, subs):
    return {x: DataLoader(dataset[x], batch_size=4, num_workers=4, shuffle=True) for x in subs}


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

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(15, 12))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)