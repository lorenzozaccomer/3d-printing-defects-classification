###
# 
#  dirs.py
#
###

import os

def ReturnDirectories(DesiredPath):
    """
    Return directories on the desired path
    """

    return os.listdir(DesiredPath)