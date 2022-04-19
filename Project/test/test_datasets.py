
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def TEST__ImagesDatasetFromFolders(path):
    '''
    Will create 2 dataset, one from train folder
    and one from valid folder
    '''
    return {x: datasets.ImageFolder(os.path.join(path, x), data_transforms[x]) for x in ['train', 'val']}


def TEST__ShuffleDatasets(datasets):
    "This function takes the datasets and shuffle them"

    """Not set num_workers because this settings can create compiling error,
    so leave the default value"""
    return {x: DataLoader(datasets[x], batch_size=4, shuffle=True) for x in ['train', 'val']}


def TEST__ImagesDatasetSize(dataset):
    '''
    Return the length of the image datasets
    '''
    return {x: len(dataset[x]) for x in ['train', 'val']}