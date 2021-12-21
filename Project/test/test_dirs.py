
import os
import cv2, torch, torchvision

path = 'D:\\repositories\\3d-printer-recognition\\Images'

# Return the folders on Image folder
def test__ReturnImageFolderFromPath(path):
    return os.listdir(path)

# Test image list
def test__ImageList(path):
    folder_list = []
    for f in os.listdir(path):
        folder_list.append(os.path.join(path,f))
    return folder_list

print("test_ImageList")
print(test__ImageList(path))

# Return the paths from primary folder
def test__FoldersOnImageList(path):
    return list((os.path.join(path,f) for f in os.listdir(path)))

print("test_ImageList")
print(test__ImageList(path))

def test__create_datasets(path):
    return {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                  for x in ['train', 'valid']}
print(test__create_datasets(path))

# show the folder inside Images folder
print('test__ReturnImageFolderFromPath')
images_folder = test__ReturnImageFolderFromPath(path)
print(images_folder)

# lists the folders on Image folder
print('test__FoldersOnImageList')
list_path_folders = test__FoldersOnImageList(path)
print(list_path_folders)
