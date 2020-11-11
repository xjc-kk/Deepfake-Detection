'''
Generate perturbed dataset.
'''
import os
import json
import cv2
import dlib
import copy
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from random import sample
from torchvision import transforms


def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    preprocessed_image = preprocess(Image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def PGD_attack(image, model, mask=None, eps=0.2, alpha=2/255, iters=20, cuda=True, is_region=False):
    """
    PGD attacks the image.
    """
    # Preprocess
    image_original = copy.deepcopy(image)

    height, width = image.shape[0], image.shape[1]

    image = preprocess_image(image, cuda)
    target = torch.tensor([1]).cuda()
    loss_f = torch.nn.CrossEntropyLoss()

    original_image = image.data

    for i in range(iters):
        image.requires_grad = True

        output = model(image)

        model.zero_grad()

        loss = loss_f(output, target)
        loss.backward()

        adv_image = image + alpha * image.grad.sign()
        eta = torch.clamp(adv_image - original_image, min=-eps, max=eps)
        image = torch.clamp(original_image + eta, min=-1, max=1).detach_()

    # denormalize perturbed image for visualization
    image = image.squeeze()
    mean = [0.5] * 3
    std = [0.5] * 3
    image[0] = image[0] * std[0] + mean[0]
    image[1] = image[1] * std[1] + mean[1]
    image[2] = image[2] * std[2] + mean[2]

    image = (image * 255).byte()
    image = image.cpu().detach().numpy().transpose((1, 2, 0))
    image = transforms.Resize((height, width))(Image.fromarray(image))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # limit the perturbation area in a certain region
    if is_region:
        image_original[np.where(mask == 1)] = 0
        image[np.where(mask == 0)] = 0
        image  = image_original + image
    
    return image


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

    src_path = '/mnt/xjc/cvpr_images'
    dst_path = '/mnt/xjc/cvpr_images'

    frame_press_rate = ['FFppDFaceC0', 'FFppDFaceC23', 'FFppDFaceC40']
    fake_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    data_types = ['train', 'val', 'test']
    models = ['c0/xception_original_c0_best.pkl', 'c23/xception_original_c23_best.pkl', 'c40/xception_original_c40_best.pkl']

    model_path = '/mnt/xjc/Deepfake-Detection/cvpr_models'

    train_list = []
    val_list = []
    test_list = []

    with open('train.json') as f:
        train = json.load(f)
        for pair in train:
            train_list.append(pair[0] + '_' + pair[1])
            train_list.append(pair[1] + '_' + pair[0])

    with open('val.json') as f:
        val = json.load(f)
        for pair in val:
            val_list.append(pair[0] + '_' + pair[1])
            val_list.append(pair[1] + '_' + pair[0])

    with open('test.json') as f:
        test = json.load(f)
        for pair in test:
            test_list.append(pair[0] + '_' + pair[1])
            test_list.append(pair[1] + '_' + pair[0])

    # c23 model, Deepfakes, global_perturbed
    pr = 'FFppDFaceC40'
    print('Processing ' + pr)
    certain_model_path = os.path.join(model_path, 'c40/xception_original_c40_best.pkl')
    model = torch.load(certain_model_path)
    model = torch.nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    frame_path = os.path.join(src_path, pr, 'manipulated')
    fake_type = 'NeuralTextures'
    print('Processing ' + fake_type)
    fake_path = os.path.join(frame_path, fake_type)
    pbar = tqdm(total=1000)
    for data_type in data_types:
        fake_data_path = os.path.join(fake_path, data_type)
        for fake_name in os.listdir(fake_data_path):
            fake_video_path = os.path.join(fake_data_path, fake_name)
            all_frames = sorted(os.listdir(fake_video_path))
            frame_num = len(all_frames)

            for img_name in all_frames:
                img_path = os.path.join(fake_video_path, img_name)
                img = cv2.imread(img_path)

                perturbed_imgdir_path = os.path.join(dst_path, pr, 'region_perturbed_landmarks20', fake_type, data_type, fake_name)
                os.makedirs(perturbed_imgdir_path, exist_ok=True)
                perturbed_img_path = os.path.join(perturbed_imgdir_path, img_name)
                if os.path.exists(perturbed_img_path):
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_detector = dlib.get_frontal_face_detector()
                faces = face_detector(gray, 1)
                # For now only take biggest face
                if len(faces):
                    face = faces[0]             
                    # face predictor
                    face_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
                    landmarks = face_predictor(gray, face)                  
                    # perturbate in a small part of the face
                    num_list = [i for i in range(68)]
                    sample_list = sorted(sample(num_list, 20))

                    points_list = []
                    for i in sample_list:
                    # for i in num_list:
                        p = landmarks.parts()[i]
                        points_list.append((p.x, p.y))
                    points = cv2.convexHull(np.array(points_list)).reshape((-1, 1, 2))

                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    mask = cv2.fillConvexPoly(mask, points, 1)

                    perturbed_img = PGD_attack(img, model, mask, iters=20, is_region=True)
                    cv2.imwrite(perturbed_img_path, perturbed_img)
            pbar.update(1)
    pbar.close()