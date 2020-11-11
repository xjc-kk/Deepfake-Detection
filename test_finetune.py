'''
test the finetuned model accuracy on original / manipulated test set
'''
from __future__ import print_function, division
import os
import time
import copy
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

test_original_path = '/mnt/xjc/cvpr_images/FFppDFaceC40/original/test'
test_Deepfakes_path = '/mnt/xjc/cvpr_images/FFppDFaceC40/region_perturbed/Deepfakes/test'
test_Face2Face_path = '/mnt/xjc/cvpr_images/FFppDFaceC40/region_perturbed/Face2Face/test'
test_FaceSwap_path = '/mnt/xjc/cvpr_images/FFppDFaceC40/region_perturbed/FaceSwap/test'
test_NeuralTextures_path ='/mnt/xjc/cvpr_images/FFppDFaceC40/region_perturbed/NeuralTextures/test'

original_path = [test_original_path]
manipulated_path = [test_Deepfakes_path, test_Face2Face_path, test_FaceSwap_path, test_NeuralTextures_path]

batch_size = 64


def get_dataset(path):
    '''
    return a list of (img_path, label) for each image
    '''
    if path in original_path:
        label = 0
    elif path in manipulated_path:
        label = 1
    else:
        print('path not valid!')
        assert 0 == 1
    
    images = []
    for video_name in os.listdir(path):
        frames_path = os.path.join(path, video_name)
        for img_name in os.listdir(frames_path):
            img_path = os.path.join(frames_path, img_name)
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


def data_loader(batch_size=40):

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

    test_original_dataloader = torch.utils.data.DataLoader(test_original_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
    test_Deepfakes_dataloader = torch.utils.data.DataLoader(test_Deepfakes_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
    test_Face2Face_dataloader = torch.utils.data.DataLoader(test_Face2Face_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
    test_FaceSwap_dataloader = torch.utils.data.DataLoader(test_FaceSwap_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
    test_NeuralTextures_dataloader = torch.utils.data.DataLoader(test_NeuralTextures_dataset, batch_size=batch_size, shuffle=True, num_workers=32)

    test_dataloader = [test_original_dataloader, test_Deepfakes_dataloader, test_Face2Face_dataloader, test_FaceSwap_dataloader, test_NeuralTextures_dataloader]

    return test_dataloader


def test_model(model):
    
    since = time.time()
    model.eval()

    test_dataloader = data_loader(batch_size=batch_size)

    acc_ls = []

    for i in range(1, 5):
        corrects = 0
        test_size = len(test_dataloader[i].dataset)
        pbar = tqdm(total=test_size)

        for images, labels in test_dataloader[i]:
            images = images.cuda()
            labels = labels.cuda()

            pbar.update(batch_size)

            with torch.set_grad_enabled(False):
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels).item()
        
        acc_ls.append(round(corrects * 100 / test_size, 2))
        pbar.close()

    time_elapsed = time.time() - since
    print('Testing complete in %.0fm %.0fs'%(time_elapsed // 60, time_elapsed % 60))

    print('Accuracy:')
    print(acc_ls)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 

    model_path = '/mnt/xjc/Deepfake-Detection/cvpr_models/c40/xception_original_c40_best.pkl'
    # model_path = '/mnt/xjc/faceforensics++_models_subset/face_detection/xception/all_c40.p'
    model = torch.load(model_path)
    model = torch.nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    test_model(model)