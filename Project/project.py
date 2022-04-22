
# Libraries
import torch
import logging

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import models
import matplotlib.pyplot as plt

# importing Functions from files
from libs.dirs import *
from libs.datasets import *
from libs.training import *
from libs.visualizer import *

cudnn.benchmark = True
plt.ion()   # interactive mode

logging.basicConfig(filename='log_image_classification.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DesiredPath = "L:\\repositories\\3d-printer-recognition\\Images"

Path, Subpaths = CheckCurrentPathAndExtractSubPaths(DesiredPath)

logging.info("-----------   NEW EXECUTION  -----------")
logging.info("Path: " + Path)
logging.info("Subpaths: " + str(Subpaths))

PATH = './generated_model.pth'
iteration = 0 #skip model generation
model_visualization = 1 #skip visualize_model

image_datasets = ImagesDatasetFromFolders(Path, Subpaths)

mixed_datasets = ShuffleDatasets(image_datasets, Subpaths)
logging.debug("len(mixed_datasets['train']): " + str(len(mixed_datasets['train'])))
logging.debug("len(mixed_datasets['valid']): " + str(len(mixed_datasets['valid'])))

dataset_sizes = ImagesDatasetSize(image_datasets, Subpaths)
logging.debug("dataset_sizes: " + str(dataset_sizes))

# Extract the class from one dataset (are equal between them)
class_names = image_datasets['train'].classes

#generate_batch_images(mixed_datasets['train'],class_names)

if iteration == 0: # I want to generate a model

    logging.info("loading model generation ..")

    model_ft = models.resnet50(pretrained=True)

    for param in model_ft.parameters():
        param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    starting_time = time.time()
    generated_model = train_model(mixed_datasets, dataset_sizes, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)
    logging.info('Training time: {:10f} minutes'.format((time.time()-starting_time)/60))
    
    logging.info("saving the model ..")
    torch.save(generated_model, PATH)
    logging.info("model saved!")
    
    if(model_visualization == 0): # only model generation
        exit()
    else:
        visualize_model(mixed_datasets, class_names, generated_model)
        plt.ioff()
        plt.show()

else:
    logging.info("skipped a new model generation ..")
    logging.info("default model loading ..")

    total = correct = 0
    
    loaded_model = torch.load(PATH)
    
    images, labels = next(iter(mixed_datasets['valid']))
    
    outputs = loaded_model(images)

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
    logging.info("show prediction images ..")
    
    imshow(torchvision.utils.make_grid(images), prediction_text)

    plt.ioff()
    plt.show()
