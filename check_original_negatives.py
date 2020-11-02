from __future__ import print_function, division
import os
import time
import json
import copy
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dataset(path):
    '''
    return a list of (img_path, label) for each image
    '''
    if path in original_path:
        label = 0
    elif path in perturbed_path:
        label = 1
    else:
        print('path not valid!')
        assert 0 == 1
    
    images = []
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        item = (img_path, label)
        images.append(item)

    return images


# My Dataset
class MyDataset(torch.utils.data.Dataset):
  
    def __init__(self, path, data_transforms=None):
        self.dataset = get_dataset(path)
        self.data_transforms = data_transforms

    def __getitem__(self, item):
        img_path, label = self.dataset[item]
        img = Image.open(img_path)
        img = self.data_transforms(img)
        return img, label

    def __len__(self):
        return len(self.dataset)


def data_loader():

    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ]),
    }

    test_original_dataset = MyDataset(path=test_original_path, data_transforms=data_transforms['test'])
    test_original_dataloader = torch.utils.data.DataLoader(test_original_dataset, batch_size=1, shuffle=True, num_workers=32)

    return test_original_dataloader

def test_model(model):

    model.eval()

    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ]),
    }

    for img_path in os.listdir('/mnt/xjc/images/original/c23/test'):
        img = Image.open(os.path.join('/mnt/xjc/images/original/c23/test', img_path))
        img = data_transforms['test'](img)
        img = img.unsqueeze(0)
        img = img.to(DEVICE)

        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        if preds.item() != 0:
            print(img_path)


if __name__ == '__main__':
    model_list = ['PGD1', 'PGD2', 'PGD3', 'PGD4', 'PGD5', 'PGD6', 'PGD7', 'PGD8', 'PGD9', 'PGD10', 'PGD20']
    for model_type in model_list:
        print('Model: %s.'%model_type)
        model_path = '/mnt/xjc/Deepfake-Detection/models/xception_' + model_type + '.pkl'
        model = torch.load(model_path)
        model = model.to(DEVICE)
        test_model(model)