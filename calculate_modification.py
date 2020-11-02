import os
import cv2
import numpy as np

manipulated_img_path = '/mnt/xjc/images/manipulated'

img_path_prefix = '/mnt/xjc/images/PGD'
img_path_suffix = '_perturbed'
PGD_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '20']
for iter_time in PGD_list:
    print('Processing iters %s'%iter_time)
    perturbation_list = []
    # images/PGD5-perturbed
    img_path = img_path_prefix + iter_time + img_path_suffix
    for fake_type in os.listdir(manipulated_img_path):
        # images/PGD5-perturbed/Deepfakes/c23
        manipulated_path = os.path.join(manipulated_img_path, fake_type, 'c23')
        perturbed_path = os.path.join(img_path, fake_type, 'c23')
        for data_type in os.listdir(manipulated_path):
            # images/PGD5-perturbed/Deepfakes/c23/train
            manipulated_images = os.path.join(manipulated_path, data_type)
            perturbed_images = os.path.join(perturbed_path, data_type)
            for path in os.listdir(perturbed_images):
                # images/PGD5-perturbed/Deepfakes/c23/train/000_001.jpg
                perturbed_im = cv2.imread(os.path.join(perturbed_images, path))
                manipulated_im = cv2.imread(os.path.join(manipulated_images, path))
                perturbation_list.append(np.mean(np.abs(perturbed_im.astype('float') - manipulated_im.astype('float'))))
    print('Iters: %s, List length: %d, Mean:%f'%(iter_time, len(perturbation_list), np.mean(perturbation_list)))
    np.save(iter_time + '_perturbation_list.npy', perturbation_list)