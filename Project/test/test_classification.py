

from PIL import Image

import torch
from torchvision import transforms, models


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 ])

dir(models)

img = Image.open("L:\\Università\\repositories\\3d-printer-recognition\\Project\\test\\dog.jpg")

img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# Load alexnet model
alexnet = models.alexnet(pretrained=True)

# Put our model in eval mode
alexnet.eval()

# Carry out inference
out = alexnet(batch_t)
print(out.shape)

# Load labels
with open('L:\\Università\\repositories\\3d-printer-recognition\\Project\\test\\imagenet_classes.txt') as f:
  classes = [line.strip() for line in f.readlines()]

values, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

for index in indices[0][:5]:
  print((classes[index], percentage[index].item()))

