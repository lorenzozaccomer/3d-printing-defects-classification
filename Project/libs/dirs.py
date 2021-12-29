
import os

#Return the paths from primary folder
def FoldersOnImageList(path):
    '''
    This function create a list of the folders inside the path folder
    ''' 
    return list((os.path.join(path,folder) for folder in ['train', 'valid']))