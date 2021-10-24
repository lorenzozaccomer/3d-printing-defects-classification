
import os

path = 'D:\\repositories\\3d-printer-recognition\\Images'

# Return the folder on Image folder
def ReturnImageFolderFromPath(path):
    return os.listdir(path)

def image_list(path):
    return (os.path.join(path,f) for f in os.listdir(path))

for f in os.listdir(path):
    print(os.path.join(path,f))