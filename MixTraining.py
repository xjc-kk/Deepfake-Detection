'''
Train an Xception model on 1) original dataset; 2) global perturbation dataset;
3) face perturbation dataset; 4) region perturbation dataset
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
train_original_path = '/mnt/xjc/images/MixTraining/train_original'
train_global_path = '/mnt/xjc/images/MixTraining/train_global'
train_face_path = '/mnt/xjc/images/MixTraining/train_face'
train_region_path = '/mnt/xjc/images/MixTraining/train_region'

# val path
val_original_path = '/mnt/xjc/images/MixTraining/val_original'
val_global_path = '/mnt/xjc/images/MixTraining/val_global'
val_face_path = '/mnt/xjc/images/MixTraining/val_face'
val_region_path = '/mnt/xjc/images/MixTraining/val_region'

# direct train path
train_original_path = '/mnt/xjc/images/DirectMixTraining/train_original'
train_global_path = '/mnt/xjc/images/DirectMixTraining/train_global'
train_face_path = '/mnt/xjc/images/DirectMixTraining/train_face'
train_region_path = '/mnt/xjc/images/DirectMixTraining/train_region'

# direct val path
val_original_path = '/mnt/xjc/images/DirectMixTraining/val_original'
val_global_path = '/mnt/xjc/images/DirectMixTraining/val_global'
val_face_path = '/mnt/xjc/images/DirectMixTraining/val_face'
val_region_path = '/mnt/xjc/images/DirectMixTraining/val_region'

num_epochs = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

original_path = [train_original_path, val_original_path]
perturbed_path = [train_global_path, train_face_path, train_region_path,
                val_global_path, val_face_path, val_region_path]


def get_dataset(path):
    '''
    return a list of (img_path, label) for each image
    '''
    # images = []
    # for img_name in os.listdir(path):
    #     img_path = os.path.join(path, img_name)
    #     label = int(img_name.split('-')[0])
    #     item = (img_path, label)
    #     images.append(item)

    # return images

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
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ]),
        'val': transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ]),
    }

    train_original_dataset = MyDataset(path=train_original_path, data_transforms=data_transforms['train'])
    train_global_dataset = MyDataset(path=train_global_path, data_transforms=data_transforms['train'])
    train_face_dataset = MyDataset(path=train_face_path, data_transforms=data_transforms['train'])
    train_region_dataset = MyDataset(path=train_region_path, data_transforms=data_transforms['train'])

    val_original_dataset = MyDataset(path=val_original_path, data_transforms=data_transforms['val'])
    val_global_dataset = MyDataset(path=val_global_path, data_transforms=data_transforms['val'])
    val_face_dataset = MyDataset(path=val_face_path, data_transforms=data_transforms['val'])
    val_region_dataset = MyDataset(path=val_region_path, data_transforms=data_transforms['val'])

    train_original_dataloader = torch.utils.data.DataLoader(train_original_dataset, batch_size=8, shuffle=True, num_workers=0)
    train_global_dataloader = torch.utils.data.DataLoader(train_global_dataset, batch_size=8, shuffle=True, num_workers=0)
    train_face_dataloader = torch.utils.data.DataLoader(train_face_dataset, batch_size=8, shuffle=True, num_workers=0)
    train_region_dataloader = torch.utils.data.DataLoader(train_region_dataset, batch_size=8, shuffle=True, num_workers=0)

    val_original_dataloader = torch.utils.data.DataLoader(val_original_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_global_dataloader = torch.utils.data.DataLoader(val_global_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_face_dataloader = torch.utils.data.DataLoader(val_face_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_region_dataloader = torch.utils.data.DataLoader(val_region_dataset, batch_size=8, shuffle=True, num_workers=0)

    train_dataloader = [train_original_dataloader, train_global_dataloader, train_face_dataloader, train_region_dataloader]
    val_dataloader = [val_original_dataloader, val_global_dataloader, val_face_dataloader, val_region_dataloader]

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

            for inputs in zip(dataloader[phase][0], dataloader[phase][1], dataloader[phase][2], dataloader[phase][3]):

                original_inputs = inputs[0][0]
                original_labels = inputs[0][1]
                global_inputs = inputs[1][0]
                global_labels = inputs[1][1]
                face_inputs = inputs[2][0]
                face_labels = inputs[2][1]
                region_inputs = inputs[3][0]
                region_labels = inputs[3][1]

                images = torch.cat((original_inputs, global_inputs, face_inputs, region_inputs), 0)
                labels = torch.cat((original_labels, global_labels, face_labels, region_labels), 0)

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
    torch.save(model, 'models/xception_DirectMixTraining.pkl')