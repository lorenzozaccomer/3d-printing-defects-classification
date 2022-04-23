
###
# 
#  visualizer.py
#
###

import torch, torchvision, logging
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


def visualize_generated_model(loaded_dataset, class_names, model):
    
    total = correct = 0

    with torch.no_grad():
        images, labels = next(iter(loaded_dataset['valid']))

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
            
        # print images
        label_text = 'Label: ' + ' '.join('%s' % class_names[x] for x in labels)
        predicted_text = 'Predicted: ' + ' '.join('%s' % class_names[predicted[j]] for j in range(images.size()[0]))
        accuracy_text = 'Prediction accuracy: {} %'.format(100 * correct / total)

        prediction_text = label_text + '\n' + predicted_text + '\n' + accuracy_text
        
        logging.info(label_text)
        logging.info(predicted_text)
        logging.info(accuracy_text)
        logging.info("show prediction ..")
        
    imshow(torchvision.utils.make_grid(images), prediction_text)

    plt.ioff()
    plt.show()       
    logging.info("end of model visualization ..")


def generate_batch_images(input_dataset, labels):

    # Get 1 batch images
    inputs, classes = next(iter(input_dataset))

    label_text = 'Label: ' + ' '.join('%s' % labels[x] for x in classes)

    # Print random images
    imshow(torchvision.utils.make_grid(inputs), label_text)


if __name__ == '__main__':
    generate_batch_images()