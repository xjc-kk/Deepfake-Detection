'''
Finetune the pretrained model on original / manipulated dataset.
'''
from __future__ import print_function, division
import os
import time
import copy
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import datasets, models, transforms

# train path
train_original_path = '/mnt/xjc/cvpr_images/FFppDFaceC0/original/train'
train_Deepfakes_path = '/mnt/xjc/cvpr_images/FFppDFaceC0/manipulated/Deepfakes/train'
train_Face2Face_path = '/mnt/xjc/cvpr_images/FFppDFaceC0/manipulated/Face2Face/train'
train_FaceSwap_path = '/mnt/xjc/cvpr_images/FFppDFaceC0/manipulated/FaceSwap/train'
train_NeuralTextures_path ='/mnt/xjc/cvpr_images/FFppDFaceC0/manipulated/NeuralTextures/train'

# val path
val_original_path = '/mnt/xjc/cvpr_images/FFppDFaceC0/original/val'
val_Deepfakes_path = '/mnt/xjc/cvpr_images/FFppDFaceC0/manipulated/Deepfakes/val'
val_Face2Face_path = '/mnt/xjc/cvpr_images/FFppDFaceC0/manipulated/Face2Face/val'
val_FaceSwap_path = '/mnt/xjc/cvpr_images/FFppDFaceC0/manipulated/FaceSwap/val'
val_NeuralTextures_path ='/mnt/xjc/cvpr_images/FFppDFaceC0/manipulated/NeuralTextures/val'

original_path = [train_original_path, val_original_path]
manipulated_path = [train_Deepfakes_path, train_Face2Face_path, train_FaceSwap_path, train_NeuralTextures_path,
                    val_Deepfakes_path, val_Face2Face_path, val_FaceSwap_path, val_NeuralTextures_path]

batch_size = 32
num_epochs = 50


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


def data_loader(batch_size=32):

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ]),
        'val': transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ]),
    }

    # 720 * 40 images for original train dataset
    train_original_dataset = MyDataset(path=train_original_path, data_transforms=data_transforms['train'])
    # 720 * 10 images for each manipulated train dataset
    train_Deepfakes_dataset = MyDataset(path=train_Deepfakes_path, data_transforms=data_transforms['train'])
    train_Face2Face_dataset = MyDataset(path=train_Face2Face_path, data_transforms=data_transforms['train'])
    train_FaceSwap_dataset = MyDataset(path=train_FaceSwap_path, data_transforms=data_transforms['train'])
    train_NeuralTextures_dataset = MyDataset(path=train_NeuralTextures_path, data_transforms=data_transforms['train'])

    # 140 * 40 images for original val dataset
    val_original_dataset = MyDataset(path=val_original_path, data_transforms=data_transforms['val'])
    # 140 * 10 images for original val dataset
    val_Deepfakes_dataset = MyDataset(path=val_Deepfakes_path, data_transforms=data_transforms['val'])
    val_Face2Face_dataset = MyDataset(path=val_Face2Face_path, data_transforms=data_transforms['val'])
    val_FaceSwap_dataset = MyDataset(path=val_FaceSwap_path, data_transforms=data_transforms['val'])
    val_NeuralTextures_dataset = MyDataset(path=val_NeuralTextures_path, data_transforms=data_transforms['val'])

    train_dataset = torch.utils.data.ConcatDataset([train_original_dataset, train_Deepfakes_dataset, train_Face2Face_dataset,
                                                                train_FaceSwap_dataset, train_NeuralTextures_dataset])
    val_dataset = torch.utils.data.ConcatDataset([val_original_dataset, val_Deepfakes_dataset, val_Face2Face_dataset,
                                                val_FaceSwap_dataset, val_NeuralTextures_dataset])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=32)

    return train_dataloader, val_dataloader

#################### modified until here ##########################

def train_model(model, criterion, optimizer, num_epochs):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    train_dataloader, val_dataloader = data_loader(batch_size=batch_size)
    dataloader = {'train':train_dataloader, 'val':val_dataloader}

    train_size, val_size = len(train_dataloader.dataset), len(val_dataloader.dataset)
    dataset_size = {'train':train_size, 'val':val_size} # 720 * 80    140 * 80
    print('Train: %d  Val: %d'%(train_size, val_size))

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

            for images, labels in dataloader[phase]:

                images = images.cuda()
                labels = labels.cuda()

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
                
                pbar.update(batch_size)
            
            pbar.close()

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects / dataset_size[phase]
            print('Loss: %.4f Acc: %.4f'%(epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

        if epoch % 10 == 9:
            print('Save best model until epoch %d'%(epoch))
            current_model_wts = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_model_wts)
            torch.save(model, 'cvpr_models/c23/xception_original_c23_%d.pkl'%(epoch+1))
            model.load_state_dict(current_model_wts)

    time_elapsed = time.time() - since
    print('Training complete in %.0fm %.0fs'%(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: %.4f'%(best_acc))

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 

    model_path = '/mnt/xjc/faceforensics++_models_subset/face_detection/xception/all_raw.p'
    model = torch.load(model_path)
    model = torch.nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8)

    model = train_model(model = model, criterion = criterion, optimizer = optimizer, num_epochs = num_epochs)
    torch.save(model, 'cvpr_models/c0/xception_original_c0_best.pkl')