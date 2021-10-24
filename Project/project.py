
# Libraries
import cv2
import os, sys

# importing Functions from files
import dirs
 
# extract the train valid folders
path = 'D:\\repositories\\3d-printer-recognition\\Images'
folders = dirs.ReturnImageFolderFromPath(path)

# show the folder inside Images folder
print(folders)

