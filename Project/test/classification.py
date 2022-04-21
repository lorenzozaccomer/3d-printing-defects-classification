

from PIL import Image
from numpy import indices

import torch
import matplotlib.pyplot as plt
from torchvision import transforms

from test_visualizer import imshow

plt.ion()   # interactive mode

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 ])

PATH_MODEL = './model_ft.pth'
PATH_IMAGE = "L:\\Università\\repositories\\3d-printer-recognition\\Project\\test\\5.jpg"
classes = ['NoDefects', 'YesDefects']

# Load model
evaluation_model = torch.load(PATH_MODEL)
evaluation_model.eval()

#Load image
img = Image.open(PATH_IMAGE)
image = transform(img)
imshow(image)


# Carry out inference
tensor_image = image.unsqueeze(0)
out = evaluation_model(tensor_image)

_, indices = torch.max(out,1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

print(classes[indices.item()], percentage[indices].item())

plt.ioff()
plt.show()