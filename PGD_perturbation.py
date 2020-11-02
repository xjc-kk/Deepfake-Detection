"""
Generates PGD attack perturbed videos for FF++ manipulated dataset

"""
import os
import argparse
from os.path import join
import cv2
import dlib
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image as pil_image
from tqdm import tqdm

from network.models import model_selection
from dataset.transform import xception_default_data_transforms
from torchvision import transforms


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


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
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output


def PGD_attack(image, model, eps=0.3, alpha=2/255, iters=5, cuda=True):
    """
    PGD attacks the image.
    """
    # Preprocess
    height, width = image.shape[0], image.shape[1]

    image = preprocess_image(image, cuda)
    target = torch.tensor([1]).cuda()
    loss_f = nn.CrossEntropyLoss()

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
    image = transforms.Resize((height, width))(pil_image.fromarray(image))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image


def test_full_image_network(video_path, model_path, output_path, cuda=True):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_name = video_path.split('/')[-1].split('.')[0]

    video_type = None
    if video_name in train_list:
        video_type = 'train'
    elif video_name in val_list:
        video_type = 'val'
    else:
        video_type = 'test'

    os.makedirs(join(output_path, 'train'), exist_ok=True)
    os.makedirs(join(output_path, 'val'), exist_ok=True)
    os.makedirs(join(output_path, 'test'), exist_ok=True)

    img_path = join(output_path, video_type)
    print('Writing images to %s.'%img_path)

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Load model
    model = torch.load(model_path)
    if isinstance(model, torch.nn.DataParallel):
      model = model.module
    if cuda:
      model = model.cuda()

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break

        # Image size
        height, width = image.shape[:2]

        # 2. Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]

            # Actual prediction using our model
            perturbed_image = PGD_attack(cropped_face, model, iters=1, cuda=cuda)
            # ------------------------------------------------------------------

            img_name = video_name + '.jpg'
            cv2.imwrite(join(output_path, video_type, img_name), perturbed_image)
            return


if __name__ == '__main__':

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

    all_video_path = [
        '/mnt/xjc/ff++/manipulated_sequences/Deepfakes/c23/videos',
        '/mnt/xjc/ff++/manipulated_sequences/Face2Face/c23/videos',
        '/mnt/xjc/ff++/manipulated_sequences/FaceSwap/c23/videos',
        '/mnt/xjc/ff++/manipulated_sequences/NeuralTextures/c23/videos'
    ]

    all_output_path = [
        '/mnt/xjc/images/PGD1_perturbed/Deepfakes/c23',
        '/mnt/xjc/images/PGD1_perturbed/Face2Face/c23',
        '/mnt/xjc/images/PGD1_perturbed/FaceSwap/c23',
        '/mnt/xjc/images/PGD1_perturbed/NeuralTextures/c23'
    ]

    model_path = '/mnt/xjc/Deepfake-Detection/models/best_epoch20.pkl'

    for i in range(4):
        video_path = all_video_path[i]
        output_path = all_output_path[i]

        cnt = 0

        if video_path.endswith('.mp4') or video_path.endswith('.avi'):
            test_full_image_network(video_path, model_path, output_path)
        else:
            videos = os.listdir(video_path)
            total = len(videos)
            for video in videos:
                cnt += 1
                test_full_image_network(join(video_path, video), model_path, output_path)
                print('Finished processing %d / %d videos.'%(cnt, total))
