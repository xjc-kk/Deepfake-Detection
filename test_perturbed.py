'''
test the model accuracy on original / perturbed test set
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
test_Deepfakes_path = '/mnt/xjc/images/PGD20_smaller_region_3alpha/Deepfakes/c23/test'
test_Face2Face_path = '/mnt/xjc/images/PGD20_smaller_region_3alpha/Face2Face/c23/test'
test_FaceSwap_path = '/mnt/xjc/images/PGD20_smaller_region_3alpha/FaceSwap/c23/test'
test_NeuralTextures_path = '/mnt/xjc/images/PGD20_smaller_region_3alpha/NeuralTextures/c23/test'

# face
# test_Deepfakes_path = '/mnt/xjc/images/PGD20_regional_perturbed/Deepfakes/c23/test'
# test_Face2Face_path = '/mnt/xjc/images/PGD20_regional_perturbed/Face2Face/c23/test'
# test_FaceSwap_path = '/mnt/xjc/images/PGD20_regional_perturbed/FaceSwap/c23/test'
# test_NeuralTextures_path = '/mnt/xjc/images/PGD20_regional_perturbed/NeuralTextures/c23/test'

# region
# test_Deepfakes_path = '/mnt/xjc/images/PGD20_smaller_region/Deepfakes/c23/test'
# test_Face2Face_path = '/mnt/xjc/images/PGD20_smaller_region/Face2Face/c23/test'
# test_FaceSwap_path = '/mnt/xjc/images/PGD20_smaller_region/FaceSwap/c23/test'
# test_NeuralTextures_path = '/mnt/xjc/images/PGD20_smaller_region/NeuralTextures/c23/test'

original_path = [test_original_path]
perturbed_path = [test_Deepfakes_path, test_Face2Face_path, test_FaceSwap_path, test_NeuralTextures_path]

batch_size = 32
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
    test_Deepfakes_dataset = MyDataset(path=test_Deepfakes_path, data_transforms=data_transforms['test'])
    test_Face2Face_dataset = MyDataset(path=test_Face2Face_path, data_transforms=data_transforms['test'])
    test_FaceSwap_dataset = MyDataset(path=test_FaceSwap_path, data_transforms=data_transforms['test'])
    test_NeuralTextures_dataset = MyDataset(path=test_NeuralTextures_path, data_transforms=data_transforms['test'])

    test_original_dataloader = torch.utils.data.DataLoader(test_original_dataset, batch_size=16, shuffle=True, num_workers=32)
    test_Deepfakes_dataloader = torch.utils.data.DataLoader(test_Deepfakes_dataset, batch_size=4, shuffle=True, num_workers=32)
    test_Face2Face_dataloader = torch.utils.data.DataLoader(test_Face2Face_dataset, batch_size=4, shuffle=True, num_workers=32)
    test_FaceSwap_dataloader = torch.utils.data.DataLoader(test_FaceSwap_dataset, batch_size=4, shuffle=True, num_workers=32)
    test_NeuralTextures_dataloader = torch.utils.data.DataLoader(test_NeuralTextures_dataset, batch_size=4, shuffle=True, num_workers=32)

    test_dataloader = [test_original_dataloader, test_Deepfakes_dataloader, test_Face2Face_dataloader, test_FaceSwap_dataloader, test_NeuralTextures_dataloader]

    return test_dataloader


def test_model(model):
    
    since = time.time()
    model.eval()

    test_dataloader = data_loader()

    test_size = 0
    for loader in test_dataloader:
        test_size += len(loader.dataset)

    pbar = tqdm(total=test_size)

    corrects_original, corrects_Deepfakes, corrects_Face2Face, corrects_FaceSwap, corrects_NeuralTextures = 0, 0, 0, 0, 0

    for inputs in zip(test_dataloader[0], test_dataloader[1], test_dataloader[2], test_dataloader[3], test_dataloader[4]):

        original_inputs = inputs[0][0]
        original_labels = inputs[0][1]
        Deepfakes_inputs = inputs[1][0]
        Deepfakes_labels = inputs[1][1]
        Face2Face_inputs = inputs[2][0]
        Face2Face_labels = inputs[2][1]
        FaceSwap_inputs = inputs[3][0]
        FaceSwap_labels = inputs[3][1]
        NeuralTextures_inputs = inputs[4][0]
        NeuralTextures_labels = inputs[4][1]

        # images = torch.cat((original_inputs, Deepfakes_inputs, Face2Face_inputs, FaceSwap_inputs, NeuralTextures_inputs), 0)
        # labels = torch.cat((original_labels, Deepfakes_labels, Face2Face_labels, FaceSwap_labels, NeuralTextures_labels), 0)

        images_original = original_inputs.to(DEVICE)
        labels_original = original_labels.to(DEVICE)

        images_Deepfakes = Deepfakes_inputs.to(DEVICE)
        labels_Deepfakes = Deepfakes_labels.to(DEVICE)

        images_Face2Face = Face2Face_inputs.to(DEVICE)
        labels_Face2Face = Face2Face_labels.to(DEVICE)

        images_FaceSwap = FaceSwap_inputs.to(DEVICE)
        labels_FaceSwap = FaceSwap_labels.to(DEVICE)

        images_NeuralTextures = NeuralTextures_inputs.to(DEVICE)
        labels_NeuralTextures = NeuralTextures_labels.to(DEVICE)

        pbar.update(batch_size)

        with torch.set_grad_enabled(False):

            outputs_original = model(images_original)
            _, preds = torch.max(outputs_original, 1)
            corrects_original += torch.sum(preds == labels_original).item()

            outputs_Deepfakes = model(images_Deepfakes)
            _, preds = torch.max(outputs_Deepfakes, 1)
            corrects_Deepfakes += torch.sum(preds == labels_Deepfakes).item()

            outputs_Face2Face = model(images_Face2Face)
            _, preds = torch.max(outputs_Face2Face, 1)
            corrects_Face2Face += torch.sum(preds == labels_Face2Face).item()

            outputs_FaceSwap = model(images_FaceSwap)
            _, preds = torch.max(outputs_FaceSwap, 1)
            corrects_FaceSwap += torch.sum(preds == labels_FaceSwap).item()

            outputs_NeuralTextures = model(images_NeuralTextures)
            _, preds = torch.max(outputs_NeuralTextures, 1)
            corrects_NeuralTextures += torch.sum(preds == labels_NeuralTextures).item()

    time_elapsed = time.time() - since
    # print('Testing complete in %.0fm %.0fs'%(time_elapsed // 60, time_elapsed % 60))

    acc_original = round(corrects_original * 100 / 560, 2)
    acc_Deepfakes = round(100 - corrects_Deepfakes * 100 / 140, 2)
    acc_Face2Face = round(100 - corrects_Face2Face * 100 / 140, 2)
    acc_FaceSwap = round(100 - corrects_FaceSwap * 100 / 140, 2)
    acc_NeuralTextures = round(100 - corrects_NeuralTextures * 100 / 140, 2)
    acc = [acc_original, acc_Deepfakes, acc_Face2Face, acc_FaceSwap, acc_NeuralTextures]
    return acc


if __name__ == '__main__':
    # model_path = '/mnt/xjc/Deepfake-Detection/models/xception_DirectMixTraining.pkl'
    model_path = '/mnt/xjc/Deepfake-Detection/models/best_epoch20.pkl'
    model = torch.load(model_path)
    model = model.to(DEVICE)

    acc = test_model(model)
    print(acc)