
import os
import cv2, torch, torchvision

# Return the folders on Image folder
def ReturnImageFolderFromPath(path):
    return os.listdir(path)

# Return the paths from primary folder
def FoldersOnImageList(path):
    return list((os.path.join(path,folder) for folder in ['train', 'valid']))

def test__create_datasets(path):
    return {x: datasets.ImageFolder(os.path.join(path, x),x)
                    for x in ['train', 'valid']}