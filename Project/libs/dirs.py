
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

    if not(os.path.isdir(DesiredPath)):
        print("path error! it not exists")
        exit()
    elif not(os.listdir(DesiredPath)):
        print("On your path there aren't folders!")
        exit()

    return DesiredPath, os.listdir(DesiredPath)