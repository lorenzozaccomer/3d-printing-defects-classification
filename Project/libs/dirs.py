
import os

def ReturnImageFolderFromPath(path):
    """
    Return the folders on Image folder
    """
    return os.listdir(path)

def FoldersOnImageList(path):
    """
    Return the paths from primary folder
    """
    return list((os.path.join(path,folder) for folder in ['train', 'valid']))

def CheckCurrentPathAndExtractSubPaths():
    """
    Check if the path that you have previously set is right,
    also return the dirs on the path,
    otherwelse it will terminate the script,
    """
    # On my computers I have 2 different paths
    Path1 = 'U:\\repositories\\3d-printer-recognition\\Images'
    Path2 = 'L:\\repositories\\3d-printer-recognition\\Images'
    DefaultPath = ""

    DefaultPath = Path1

    if not(os.path.isdir(Path1)):
        DefaultPath = Path2
    elif not(os.path.isdir(Path1) or os.path.isdir(Path2)):
        print("path error! check it")
        exit()

    return DefaultPath, os.listdir(DefaultPath)