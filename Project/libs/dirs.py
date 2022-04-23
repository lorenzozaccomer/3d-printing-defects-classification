
###
# 
#  dirs.py
#
###

import os


def CheckCurrentPathAndExtractSubPaths(DesiredPath):
    """
    Check if the path that you have previously set is right,
    also return the directories on the path,
    otherwelse it will terminate the script,
    """

    if not(os.path.isdir(DesiredPath)):
        print("path error! it not exists")
        exit()
    elif not(os.listdir(DesiredPath)):
        print("On your path there aren't folders!")
        exit()

    return DesiredPath, os.listdir(DesiredPath)