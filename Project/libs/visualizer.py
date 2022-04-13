
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


def visualize_model(dataloaders, class_names, model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['valid']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def generate_batch_images(dataloaders, class_names):

    # Get 1 batch images
    inputs, classes = next(iter(dataloaders['train']))

    # Get a batch of training data
    out = torchvision.utils.make_grid(inputs)

    print("printing..")

    # Print 4 random images
    imshow(out, title=[class_names[x] for x in classes])

    print("end showing images..")


if __name__ == '__main__':
    generate_batch_images()