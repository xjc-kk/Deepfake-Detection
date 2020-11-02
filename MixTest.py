'''
Test the model accuracy on 1) original; 2) global perturbation dataset;
3) face perturbation dataset; 4) region perturbation dataset
'''
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


test_original_path = '/mnt/xjc/images/original/c23/test'
# global
test_Deepfakes_path = '/mnt/xjc/images/PGD20_perturbed/Deepfakes/c23/test'
test_Face2Face_path = '/mnt/xjc/images/PGD20_perturbed/Face2Face/c23/test'
test_FaceSwap_path = '/mnt/xjc/images/PGD20_perturbed/FaceSwap/c23/test'
test_NeuralTextures_path = '/mnt/xjc/images/PGD20_perturbed/NeuralTextures/c23/test'

# face
test_Deepfakes_path = '/mnt/xjc/images/PGD20_regional_perturbed/Deepfakes/c23/test'
test_Face2Face_path = '/mnt/xjc/images/PGD20_regional_perturbed/Face2Face/c23/test'
test_FaceSwap_path = '/mnt/xjc/images/PGD20_regional_perturbed/FaceSwap/c23/test'
test_NeuralTextures_path = '/mnt/xjc/images/PGD20_regional_perturbed/NeuralTextures/c23/test'

# region
test_Deepfakes_path = '/mnt/xjc/images/PGD20_smaller_region/Deepfakes/c23/test'
test_Face2Face_path = '/mnt/xjc/images/PGD20_smaller_region/Face2Face/c23/test'
test_FaceSwap_path = '/mnt/xjc/images/PGD20_smaller_region/FaceSwap/c23/test'
test_NeuralTextures_path = '/mnt/xjc/images/PGD20_smaller_region/NeuralTextures/c23/test'

original_path = [test_original_path]
perturbed_path = [test_Deepfakes_path, test_Face2Face_path, test_FaceSwap_path, test_NeuralTextures_path]

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
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ]),
    }

    test_original_dataset = MyDataset(path=test_original_path, data_transforms=data_transforms['test'])
    test_Deepfakes_dataset = MyDataset(path=test_Deepfakes_path, data_transforms=data_transforms['test'])
    test_Face2Face_dataset = MyDataset(path=test_Face2Face_path, data_transforms=data_transforms['test'])
    test_FaceSwap_dataset = MyDataset(path=test_FaceSwap_path, data_transforms=data_transforms['test'])
    test_NeuralTextures_dataset = MyDataset(path=test_NeuralTextures_path, data_transforms=data_transforms['test'])

    test_original_dataloader = torch.utils.data.DataLoader(test_original_dataset, batch_size=1, shuffle=True, num_workers=32)
    test_Deepfakes_dataloader = torch.utils.data.DataLoader(test_Deepfakes_dataset, batch_size=1, shuffle=True, num_workers=32)
    test_Face2Face_dataloader = torch.utils.data.DataLoader(test_Face2Face_dataset, batch_size=1, shuffle=True, num_workers=32)
    test_FaceSwap_dataloader = torch.utils.data.DataLoader(test_FaceSwap_dataset, batch_size=1, shuffle=True, num_workers=32)
    test_NeuralTextures_dataloader = torch.utils.data.DataLoader(test_NeuralTextures_dataset, batch_size=1, shuffle=True, num_workers=32)

    test_dataloader = [test_original_dataloader, test_Deepfakes_dataloader, test_Face2Face_dataloader, test_FaceSwap_dataloader, test_NeuralTextures_dataloader]

    return test_dataloader


def test_model(model):
    
    since = time.time()
    model.eval()

    test_dataloader = data_loader()

    test_size = 0
    for loader in test_dataloader:
        test_size += len(loader.dataset)

    # pbar = tqdm(total=test_size)

    corrects_original, corrects_Deepfakes, corrects_Face2Face, corrects_FaceSwap, corrects_NeuralTextures = 0, 0, 0, 0, 0

    # test on original dataset
    for inputs, labels in tqdm(test_dataloader[0]):

        # inputs (1, 3, 300, 300)
        bs, c, h, w = inputs.shape
        patch_h, patch_w = int(h / 2), int(w / 2)
        patch1 = inputs[:, :, :patch_h, :patch_w]
        patch2 = inputs[:, :, :patch_h, patch_w:]
        patch3 = inputs[:, :, patch_h:, :patch_w]
        patch4 = inputs[:, :, patch_h:, patch_w:]

        inputs = torch.cat((patch1, patch2, patch3, patch4), 0)

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            outputs = model(inputs) # (4, 2)
            values = []
            for i in range(4):
                values.append(np.exp(outputs[i][1]) / (np.exp(outputs[i][0]) + np.exp(outputs[i][1])))
                
            # products = 1
            # for value in values:
            #     products *= (1 - value.item())
            # probability = 1 - products
            probability = np.mean(values)
            if probability >= 0.5:
                pred = 1
            else:
                pred = 0
            if pred == labels.item():
                corrects_original += 1

    # test on Deepfakes dataset
    for inputs, labels in tqdm(test_dataloader[1]):

        # inputs (1, 3, 300, 300)
        bs, c, h, w = inputs.shape
        patch_h, patch_w = int(h / 2), int(w / 2)
        patch1 = inputs[:, :, :patch_h, :patch_w]
        patch2 = inputs[:, :, :patch_h, patch_w:]
        patch3 = inputs[:, :, patch_h:, :patch_w]
        patch4 = inputs[:, :, patch_h:, patch_w:]

        inputs = torch.cat((patch1, patch2, patch3, patch4), 0)

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            outputs = model(inputs) # (4, 2)
            values = []
            for i in range(4):
                values.append(np.exp(outputs[i][1]) / (np.exp(outputs[i][0]) + np.exp(outputs[i][1])))
                
            # products = 1
            # for value in values:
            #     products *= (1 - value.item())
            # probability = 1 - products
            probability = np.mean(values)
            if probability >= 0.5:
                pred = 1
            else:
                pred = 0
            if pred == labels.item():
                corrects_Deepfakes += 1

    # test on Face2Face dataset
    for inputs, labels in tqdm(test_dataloader[2]):

        # inputs (1, 3, 300, 300)
        bs, c, h, w = inputs.shape
        patch_h, patch_w = int(h / 2), int(w / 2)
        patch1 = inputs[:, :, :patch_h, :patch_w]
        patch2 = inputs[:, :, :patch_h, patch_w:]
        patch3 = inputs[:, :, patch_h:, :patch_w]
        patch4 = inputs[:, :, patch_h:, patch_w:]

        inputs = torch.cat((patch1, patch2, patch3, patch4), 0)

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            outputs = model(inputs) # (4, 2)
            values = []
            for i in range(4):
                values.append(np.exp(outputs[i][1]) / (np.exp(outputs[i][0]) + np.exp(outputs[i][1])))
                
            # products = 1
            # for value in values:
            #     products *= (1 - value.item())
            # probability = 1 - products
            probability = np.mean(values)
            if probability >= 0.5:
                pred = 1
            else:
                pred = 0
            if pred == labels.item():
                corrects_Face2Face += 1
    
    # test on FaceSwap dataset
    for inputs, labels in tqdm(test_dataloader[3]):

        # inputs (1, 3, 300, 300)
        bs, c, h, w = inputs.shape
        patch_h, patch_w = int(h / 2), int(w / 2)
        patch1 = inputs[:, :, :patch_h, :patch_w]
        patch2 = inputs[:, :, :patch_h, patch_w:]
        patch3 = inputs[:, :, patch_h:, :patch_w]
        patch4 = inputs[:, :, patch_h:, patch_w:]

        inputs = torch.cat((patch1, patch2, patch3, patch4), 0)

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            outputs = model(inputs) # (4, 2)
            values = []
            for i in range(4):
                values.append(np.exp(outputs[i][1]) / (np.exp(outputs[i][0]) + np.exp(outputs[i][1])))
                
            # products = 1
            # for value in values:
            #     products *= (1 - value.item())
            # probability = 1 - products
            probability = np.mean(values)
            if probability >= 0.5:
                pred = 1
            else:
                pred = 0
            if pred == labels.item():
                corrects_FaceSwap += 1
    
    # test on original dataset
    for inputs, labels in tqdm(test_dataloader[4]):

        # inputs (1, 3, 300, 300)
        bs, c, h, w = inputs.shape
        patch_h, patch_w = int(h / 2), int(w / 2)
        patch1 = inputs[:, :, :patch_h, :patch_w]
        patch2 = inputs[:, :, :patch_h, patch_w:]
        patch3 = inputs[:, :, patch_h:, :patch_w]
        patch4 = inputs[:, :, patch_h:, patch_w:]

        inputs = torch.cat((patch1, patch2, patch3, patch4), 0)

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            outputs = model(inputs) # (4, 2)
            values = []
            for i in range(4):
                values.append(np.exp(outputs[i][1]) / (np.exp(outputs[i][0]) + np.exp(outputs[i][1])))
                
            # products = 1
            # for value in values:
            #     products *= (1 - value.item())
            # probability = 1 - products
            probability = np.mean(values)
            if probability >= 0.5:
                pred = 1
            else:
                pred = 0
            if pred == labels.item():
                corrects_NeuralTextures += 1
    
    time_elapsed = time.time() - since
    # print('Testing complete in %.0fm %.0fs'%(time_elapsed // 60, time_elapsed % 60))

    acc_original = round(corrects_original * 100 / 560, 2)
    acc_Deepfakes = round(corrects_Deepfakes * 100 / 140, 2)
    acc_Face2Face = round(corrects_Face2Face * 100 / 140, 2)
    acc_FaceSwap = round(corrects_FaceSwap * 100 / 140, 2)
    acc_NeuralTextures = round(corrects_NeuralTextures * 100 / 140, 2)
    acc = [acc_original, acc_Deepfakes, acc_Face2Face, acc_FaceSwap, acc_NeuralTextures]
    return acc


if __name__ == '__main__':

    model_path = '/mnt/xjc/Deepfake-Detection/models/xception_MixTraining.pkl'
    model = torch.load(model_path)
    model = model.to(DEVICE)

    acc = test_model(model)
    print(acc)