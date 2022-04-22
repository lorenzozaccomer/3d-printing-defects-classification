
###
# 
#  visualizer.py
#
###

import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np

from libs.constants import *

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(15, 12))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def visualize_model(loaded_dataset, class_names, model):

    with torch.no_grad():
        images, labels = next(iter(loaded_dataset['valid']))

        # print images
        imshow(torchvision.utils.make_grid(images), title=[class_names[x] for x in labels])

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join('%s' % class_names[predicted[j]] for j in range(images.size()[0])))           


def generate_batch_images(input_dataset, class_names):

    # Get 1 batch images
    inputs, classes = next(iter(input_dataset))

    # Print random images
    imshow(torchvision.utils.make_grid(inputs), title=[class_names[x] for x in classes])


if __name__ == '__main__':
    generate_batch_images()