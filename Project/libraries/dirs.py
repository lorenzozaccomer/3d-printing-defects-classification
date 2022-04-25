
###
# 
#  dirs.py
#
###

import os
import logging


def CheckDirectories(DesiredPath):
    """
    Check if on the path there are directories,
    otherwelse it will terminate the script,
    """

    if not(os.listdir(DesiredPath)):
        print("On your path there aren't folders!")
        logging.debug("NOT DIRECTORIES ERROR")
        exit()

    return os.listdir(DesiredPath)