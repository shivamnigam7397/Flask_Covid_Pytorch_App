import torch, torchvision
import numpy as np
import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
import time
import copy
import os
import io


def load_model(checkpoint_path):
    chpt = torch.load(checkpoint_path)

    if chpt['arch'] == 'vgg16_bn':
      model = models.vgg16_bn(pretrained=True)

    model.class_to_idx = chpt['class_to_idx']
    model.class_names = chpt['class_names']


    # Put the classifier on the pretrained network
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 2)

    model.load_state_dict(chpt['state_dict'])

    return model

model_new = load_model("vgg16_bn_covid_nojitter_nowtdecay.pth")
print(model_new.class_names)
model_new.cpu()
model_new.eval()

loader = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
def prediction_image(model,name_image):
    image = Image.open(io.BytesIO(name_image))
    img = loader(image).unsqueeze_(0)
    print(img.shape)
    outputs = model(img)
    _, pred = torch.max(outputs.data, 1)
    pred_scalar = pred.item()
    return model.class_names[pred_scalar]


#
# def get_prediction(image_bytes):
#     try:
#         tensor = transform_image(image_bytes=image_bytes)
#         outputs = model.forward(tensor)
#     except Exception:
#         return 0, 'error'
#     _, y_hat = outputs.max(1)
#     predicted_idx = str(y_hat.item())
#     return imagenet_class_index[predicted_idx]
