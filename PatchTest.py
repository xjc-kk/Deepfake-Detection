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
from PatchTraining import CNN


test_original_path = '/mnt/xjc/images/DirectMixTraining/test_original'
# global
test_Deepfakes_path = '/mnt/xjc/images/DirectMixTraining/test_global_Deepfakes'
test_Face2Face_path = '/mnt/xjc/images/DirectMixTraining/test_global_Face2Face'
test_FaceSwap_path = '/mnt/xjc/images/DirectMixTraining/test_global_FaceSwap'
test_NeuralTextures_path = '/mnt/xjc/images/DirectMixTraining/test_global_NeuralTextures'

# # face
# test_Deepfakes_path = '/mnt/xjc/images/DirectMixTraining/test_face_Deepfakes'
# test_Face2Face_path = '/mnt/xjc/images/DirectMixTraining/test_face_Deepfakes'
# test_FaceSwap_path = '/mnt/xjc/images/DirectMixTraining/test_face_FaceSwap'
# test_NeuralTextures_path = '/mnt/xjc/images/DirectMixTraining/test_face_NeuralTextures'

# region
# test_Deepfakes_path = '/mnt/xjc/images/DirectMixTraining/test_region_Deepfakes'
# test_Face2Face_path = '/mnt/xjc/images/DirectMixTraining/test_region_Face2Face'
# test_FaceSwap_path = '/mnt/xjc/images/DirectMixTraining/test_region_FaceSwap'
# test_NeuralTextures_path = '/mnt/xjc/images/DirectMixTraining/test_region_NeuralTextures'

original_path = [test_original_path]
perturbed_path = [test_Deepfakes_path, test_Face2Face_path, test_FaceSwap_path, test_NeuralTextures_path]

batch_size = 1
patch_num = 3
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
        patch_h, patch_w = int(h / patch_num), int(w / patch_num)
        patches = []
        for i in range(patch_num):
            for j in range(patch_num):
                patches.append(inputs[:, :, i*patch_h:(i + 1)*patch_h, j*patch_w:(j + 1)*patch_w])

        inputs = torch.cat(([patch for patch in patches]), 0) # (9, 3, 100, 100)

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            outputs = model(inputs) # (1, 2)
            _, preds = torch.max(outputs, 1)
            corrects_original += torch.sum(preds == labels).item()

    # test on Deepfakes dataset
    for inputs, labels in tqdm(test_dataloader[1]):

        # inputs (1, 3, 300, 300)
        bs, c, h, w = inputs.shape
        patch_h, patch_w = int(h / patch_num), int(w / patch_num)
        patches = []
        for i in range(patch_num):
            for j in range(patch_num):
                patches.append(inputs[:, :, i*patch_h:(i + 1)*patch_h, j*patch_w:(j + 1)*patch_w])

        inputs = torch.cat(([patch for patch in patches]), 0) # (16, 3, 75, 75)

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            outputs = model(inputs) # (1, 2)
            _, preds = torch.max(outputs, 1)
            corrects_Deepfakes += torch.sum(preds == labels).item()

    # test on Face2Face dataset
    for inputs, labels in tqdm(test_dataloader[2]):

        # inputs (1, 3, 300, 300)
        bs, c, h, w = inputs.shape
        patch_h, patch_w = int(h / patch_num), int(w / patch_num)
        patches = []
        for i in range(patch_num):
            for j in range(patch_num):
                patches.append(inputs[:, :, i*patch_h:(i + 1)*patch_h, j*patch_w:(j + 1)*patch_w])

        inputs = torch.cat(([patch for patch in patches]), 0) # (16, 3, 75, 75)

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            outputs = model(inputs) # (1, 2)
            _, preds = torch.max(outputs, 1)
            corrects_Face2Face += torch.sum(preds == labels).item()
    
    # test on FaceSwap dataset
    for inputs, labels in tqdm(test_dataloader[3]):

        # inputs (1, 3, 300, 300)
        bs, c, h, w = inputs.shape
        patch_h, patch_w = int(h / patch_num), int(w / patch_num)
        patches = []
        for i in range(patch_num):
            for j in range(patch_num):
                patches.append(inputs[:, :, i*patch_h:(i + 1)*patch_h, j*patch_w:(j + 1)*patch_w])

        inputs = torch.cat(([patch for patch in patches]), 0) # (16, 3, 75, 75)

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            outputs = model(inputs) # (1, 2)
            _, preds = torch.max(outputs, 1)
            corrects_FaceSwap += torch.sum(preds == labels).item()
    
    # test on original dataset
    for inputs, labels in tqdm(test_dataloader[4]):

        # inputs (1, 3, 300, 300)
        bs, c, h, w = inputs.shape
        patch_h, patch_w = int(h / patch_num), int(w / patch_num)
        patches = []
        for i in range(patch_num):
            for j in range(patch_num):
                patches.append(inputs[:, :, i*patch_h:(i + 1)*patch_h, j*patch_w:(j + 1)*patch_w])

        inputs = torch.cat(([patch for patch in patches]), 0) # (16, 3, 75, 75)

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            outputs = model(inputs) # (1, 2)
            _, preds = torch.max(outputs, 1)
            corrects_NeuralTextures += torch.sum(preds == labels).item()
    
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

    model_path = '/mnt/xjc/Deepfake-Detection/models/xception_PatchTraining_9.pkl'
    model = torch.load(model_path)
    model = model.to(DEVICE)

    acc = test_model(model)
    print(acc)