
###
# 
#  dirs.py
#
###

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


def CheckCurrentPathAndExtractSubPaths(DesiredPath):
    """
    Check if the path that you have previously set is right,
    also return the dirs on the path,
    otherwelse it will terminate the script,
    """
    # On my computers I have 2 different paths
    #Path1 = 'U:\\repositories\\3d-printer-recognition\\Images'
    Path2 = 'L:\\Università\\repositories\\3d-printer-recognition\\Images'
    PathVM = '/home/lorenzo/Scrivania/3d-printer-recognition/Images'
    DefaultPath = ""

    DefaultPath = DesiredPath

    # for VM Paths
    if os.path.isdir(PathVM):
        DefaultPath = PathVM

    # for My Windows PCs
    if not(os.path.isdir(DefaultPath)):
        DefaultPath = Path2
    elif not(os.path.isdir(DefaultPath) or os.path.isdir(Path2)):
        print("path error! check it")
        exit()

    if not(os.listdir(DefaultPath)):
        print("not directories on this path")
        exit()

    return DefaultPath, os.listdir(DefaultPath)