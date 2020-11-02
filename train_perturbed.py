'''
train an Xception model to classify original images and perturbed images
original as 0
perturbed as 1
'''
from __future__ import print_function, division
import os
import time
import copy
import torch
import numpy as np
import pretrainedmodels
from PIL import Image
from tqdm import tqdm
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

# train path
train_original_path = '/mnt/xjc/images/original/c23/train'
train_Deepfakes_path = '/mnt/xjc/images/PGD1-5_regional_perturbed/Deepfakes/c23/train'
train_Face2Face_path = '/mnt/xjc/images/PGD1-5_regional_perturbed/Face2Face/c23/train'
train_FaceSwap_path = '/mnt/xjc/images/PGD1-5_regional_perturbed/FaceSwap/c23/train'
train_NeuralTextures_path ='/mnt/xjc/images/PGD1-5_regional_perturbed/NeuralTextures/c23/train'

# val path
val_original_path = '/mnt/xjc/images/original/c23/val'
val_Deepfakes_path = '/mnt/xjc/images/PGD1-5_regional_perturbed/Deepfakes/c23/val'
val_Face2Face_path = '/mnt/xjc/images/PGD1-5_regional_perturbed/Face2Face/c23/val'
val_FaceSwap_path = '/mnt/xjc/images/PGD1-5_regional_perturbed/FaceSwap/c23/val'
val_NeuralTextures_path ='/mnt/xjc/images/PGD1-5_regional_perturbed/NeuralTextures/c23/val'

original_path = [train_original_path, val_original_path]
perturbed_path = [train_Deepfakes_path, train_Face2Face_path, train_FaceSwap_path, train_NeuralTextures_path,
                    val_Deepfakes_path, val_Face2Face_path, val_FaceSwap_path, val_NeuralTextures_path]


batch_size = 24
num_epochs = 5
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
        'train': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ]),
        'val': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ]),
    }

    train_original_dataset = MyDataset(path=train_original_path, data_transforms=data_transforms['train'])
    train_Deepfakes_dataset = MyDataset(path=train_Deepfakes_path, data_transforms=data_transforms['train'])
    train_Face2Face_dataset = MyDataset(path=train_Face2Face_path, data_transforms=data_transforms['train'])
    train_FaceSwap_dataset = MyDataset(path=train_FaceSwap_path, data_transforms=data_transforms['train'])
    train_NeuralTextures_dataset = MyDataset(path=train_NeuralTextures_path, data_transforms=data_transforms['train'])

    val_original_dataset = MyDataset(path=val_original_path, data_transforms=data_transforms['val'])
    val_Deepfakes_dataset = MyDataset(path=val_Deepfakes_path, data_transforms=data_transforms['val'])
    val_Face2Face_dataset = MyDataset(path=val_Face2Face_path, data_transforms=data_transforms['val'])
    val_FaceSwap_dataset = MyDataset(path=val_FaceSwap_path, data_transforms=data_transforms['val'])
    val_NeuralTextures_dataset = MyDataset(path=val_NeuralTextures_path, data_transforms=data_transforms['val'])

    train_original_dataloader = torch.utils.data.DataLoader(train_original_dataset, batch_size=4, shuffle=True, num_workers=0)
    train_Deepfakes_dataloader = torch.utils.data.DataLoader(train_Deepfakes_dataset, batch_size=5, shuffle=True, num_workers=0)
    train_Face2Face_dataloader = torch.utils.data.DataLoader(train_Face2Face_dataset, batch_size=5, shuffle=True, num_workers=0)
    train_FaceSwap_dataloader = torch.utils.data.DataLoader(train_FaceSwap_dataset, batch_size=5, shuffle=True, num_workers=0)
    train_NeuralTextures_dataloader = torch.utils.data.DataLoader(train_NeuralTextures_dataset, batch_size=5, shuffle=True, num_workers=0)

    val_original_dataloader = torch.utils.data.DataLoader(val_original_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_Deepfakes_dataloader = torch.utils.data.DataLoader(val_Deepfakes_dataset, batch_size=5, shuffle=True, num_workers=0)
    val_Face2Face_dataloader = torch.utils.data.DataLoader(val_Face2Face_dataset, batch_size=5, shuffle=True, num_workers=0)
    val_FaceSwap_dataloader = torch.utils.data.DataLoader(val_FaceSwap_dataset, batch_size=5, shuffle=True, num_workers=0)
    val_NeuralTextures_dataloader = torch.utils.data.DataLoader(val_NeuralTextures_dataset, batch_size=5, shuffle=True, num_workers=0)

    train_dataloader = [train_original_dataloader, train_Deepfakes_dataloader, train_Face2Face_dataloader, train_FaceSwap_dataloader, train_NeuralTextures_dataloader]
    val_dataloader = [val_original_dataloader, val_Deepfakes_dataloader, val_Face2Face_dataloader, val_FaceSwap_dataloader, val_NeuralTextures_dataloader]

    return train_dataloader, val_dataloader

#################### modified until here ##########################

def train_model(model, criterion, scheduler, optimizer, num_epochs):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    train_dataloader, val_dataloader = data_loader()
    dataloader = {'train':train_dataloader, 'val':val_dataloader}

    train_size, val_size = 0, 0
    for loader in train_dataloader:
        train_size += len(loader.dataset)
    for loader in val_dataloader:
        val_size += len(loader.dataset)
    dataset_size = {'train':train_size, 'val':val_size}

    for epoch in range(num_epochs):

        print('Epoch %d/%d'%(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            pbar = tqdm(total=dataset_size[phase])

            for inputs in zip(dataloader[phase][0], dataloader[phase][1], dataloader[phase][2], dataloader[phase][3], dataloader[phase][4]):

                original_inputs = inputs[0][0]
                original_labels = inputs[0][1]
                Deepfakes_inputs = inputs[1][0]
                Deepfakes_labels = inputs[1][1]
                Face2Face_inputs = inputs[2][0]
                Face2Face_labels = inputs[2][1]
                FaceSwap_inputs = inputs[3][0]
                FaceSwap_labels = inputs[3][1]
                NeuralTextures_input = inputs[4][0]
                NeuralTextures_labels = inputs[4][1]

                images = torch.cat((original_inputs, Deepfakes_inputs, Face2Face_inputs, FaceSwap_inputs, NeuralTextures_input), 0)
                labels = torch.cat((original_labels, Deepfakes_labels, Face2Face_labels, FaceSwap_labels, NeuralTextures_labels), 0)

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                pbar.update(images.shape[0])

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)   
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data).item()
            
            pbar.close()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects / dataset_size[phase]
            print('Loss: %.4f Acc: %.4f'%(epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in %.0fm %.0fs'%(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: %.4f'%(best_acc))

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':

    # pretrained Xception model
    model = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')
    in_features = model.last_linear.in_features
    num_classes = 2
    model.last_linear = torch.nn.Linear(in_features, num_classes)
    model = model.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    model = train_model(model = model, criterion = criterion, scheduler = scheduler, optimizer = optimizer, num_epochs = num_epochs)
    torch.save(model, 'models/xception_PGD1-5_regional.pkl')