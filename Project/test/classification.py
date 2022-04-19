

from PIL import Image
from numpy import indices

import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 ])

PATH = './model_ft.pth'

img = Image.open("L:\\Università\\repositories\\3d-printer-recognition\\Project\\test\\6.jpg")

#img.show()

img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# Load alexnet model
evaluation_model = torch.load(PATH)

# Put our model in eval mode
evaluation_model.eval()

# Carry out inference
out = evaluation_model(batch_t)
print(out.shape)

# Load labels
with open('L:\\Università\\repositories\\3d-printer-recognition\\Project\\test\\3d_classes.txt') as f:
  classes = [line.strip() for line in f.readlines()]

values, indices = torch.max(out,1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

for index in indices[:2]:
  print((classes[index], percentage[index].item()))