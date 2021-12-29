
import torch, torchvision
import os

from torchvision import datasets

def ImagesDatasetFromFolders(path, subs):
    '''
    Will create 2 dataset, one from train folder
    and one from valid folder
    '''
    return {x: datasets.ImageFolder(os.path.join(path, x), x)
                  for x in subs}

def ImagesDatasetSize(dataset, subs):
    '''
    Return the length of the image datasets
    '''
    return {x: len(dataset[x]) for x in subs}