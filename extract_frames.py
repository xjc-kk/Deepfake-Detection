"""
Extract frames from FF++ videos.
270 frames for train set, 100 frames for val/test set.
"""
import os
import argparse
from os.path import join
import json
import cv2
import dlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as pil_image
from tqdm import tqdm

from network.models import model_selection
from dataset.transform import xception_default_data_transforms


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


def extract_frames(video_path, output_path, cuda=True):
    print('Starting: {}'.format(video_path))

    video_name = video_path.split('/')[-1].split('.')[0]
    # first_video_name = video_name.split('_')[0]

    # Read
    reader = cv2.VideoCapture(video_path)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_name in train_list:
        output_path = '/mnt/xjc/images/DirectMixTraining/train_original_more'
    else:
        if video_name in val_list:
            output_path = '/mnt/xjc/images/DirectMixTraining/val_original_more'
        else:
            return

    # extract 270 frames for train videos and 100 frames for val/test videos
    # interval = 1
    # sample_frames = []
    # video_type = None
    # if first_video_name in train_list:
    #     video_type = 'train'
    #     interval = int(num_frames / 270)
    #     sample_frames = [1 + i * interval for i in range(270)]
    # else:
    #     if first_video_name in val_list:
    #         video_type = 'val'
    #     else:
    #         video_type = 'test'
    #     interval = int(num_frames / 100)
    #     sample_frames = [1 + i * interval for i in range(100)]

    # os.makedirs(join(output_path, 'train'), exist_ok=True)
    # os.makedirs(join(output_path, 'val'), exist_ok=True)
    # os.makedirs(join(output_path, 'test'), exist_ok=True)

    # img_path = join(output_path, video_type)
    # print('Writing images to %s.'%img_path)

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    frame_num = 0

    sample_frames = [5 * i for i in range(1, 13)]

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break

        # Image size
        height, width = image.shape[:2]

        # Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            
            frame_num += 1
            # For now only take biggest face
            face = faces[0]
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]

            if frame_num in sample_frames:
                img_name = video_name + '_frame' + str(frame_num) + '.jpg'
                cv2.imwrite(join(output_path, img_name), cropped_face)
            
            if frame_num == 80:
                return


if __name__ == '__main__':

    train_list = []
    val_list = []
    test_list = []

    with open('train.json') as f:
        train = json.load(f)
        for pair in train:
            train_list.append(pair[0])
            train_list.append(pair[1])

    with open('val.json') as f:
        val = json.load(f)
        for pair in val:
            val_list.append(pair[0])
            val_list.append(pair[1])

    with open('test.json') as f:
        test = json.load(f)
        for pair in test:
            test_list.append(pair[0])
            test_list.append(pair[1])
    
    video_path = '/mnt/xjc/ff++/original_sequences/youtube/c23/videos'
    output_path = '/mnt/xjc/images/DirectMixTraining/train_original_more'

    cnt = 0

    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        extract_frames(video_path, output_path)
    else:
        videos = os.listdir(video_path)
        total = len(videos)
        for video in videos:
            cnt += 1
            extract_frames(join(video_path, video), output_path)
            print('Finished processing %d / %d videos.'%(cnt, total))
    # all_video_path = [
    #     '/mnt/xjc/ff++/manipulated_sequences/Deepfakes/c23/videos',
    #     '/mnt/xjc/ff++/manipulated_sequences/Face2Face/c23/videos',
    #     '/mnt/xjc/ff++/manipulated_sequences/FaceSwap/c23/videos',
    #     '/mnt/xjc/ff++/manipulated_sequences/NeuralTextures/c23/videos'
    # ]

    # all_output_path = [
    #     '/mnt/xjc/images/manipulated/Deepfakes/c23',
    #     '/mnt/xjc/images/manipulated/Face2Face/c23',
    #     '/mnt/xjc/images/manipulated/FaceSwap/c23',
    #     '/mnt/xjc/images/manipulated/NeuralTextures/c23'
    # ]
    
    # for i in range(4):
    #     video_path = all_video_path[i]
    #     output_path = all_output_path[i]

    #     cnt = 0

    #     if video_path.endswith('.mp4') or video_path.endswith('.avi'):
    #         extract_frames(video_path, output_path)
    #     else:
    #         videos = os.listdir(video_path)
    #         total = len(videos)
    #         for video in videos:
    #             cnt += 1
    #             extract_frames(join(video_path, video), output_path)
    #             print('Finished processing %d / %d videos.'%(cnt, total))