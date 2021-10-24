
import os

# Return the folders on Image folder
def ReturnImageFolderFromPath(path):
    return os.listdir(path)

# Return the paths from primary folder
def FoldersOnImageList(path):
    return list((os.path.join(path,folder) for folder in ['train', 'valid']))