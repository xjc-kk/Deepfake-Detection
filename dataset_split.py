'''
split ff++ dataset based on train/val/test json file
'''
import os
import json
import shutil

########################## original #################################
os.makedirs('/mnt/xjc/images/original/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/original/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/original/c23/test', exist_ok=True)
######################################################################

########################## manipulated ##############################
os.makedirs('/mnt/xjc/images/manipulated/Deepfakes/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/manipulated/Deepfakes/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/manipulated/Deepfakes/c23/test', exist_ok=True)

os.makedirs('/mnt/xjc/images/manipulated/Face2Face/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/manipulated/Face2Face/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/manipulated/Face2Face/c23/test', exist_ok=True)

os.makedirs('/mnt/xjc/images/manipulated/FaceSwap/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/manipulated/FaceSwap/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/manipulated/FaceSwap/c23/test', exist_ok=True)

os.makedirs('/mnt/xjc/images/manipulated/NeuralTextures/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/manipulated/NeuralTextures/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/manipulated/NeuralTextures/c23/test', exist_ok=True)
######################################################################

########################## FGSM #################################
os.makedirs('/mnt/xjc/images/FGSM_perturbed/Deepfakes/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/FGSM_perturbed/Deepfakes/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/FGSM_perturbed/Deepfakes/c23/test', exist_ok=True)

os.makedirs('/mnt/xjc/images/FGSM_perturbed/Face2Face/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/FGSM_perturbed/Face2Face/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/FGSM_perturbed/Face2Face/c23/test', exist_ok=True)

os.makedirs('/mnt/xjc/images/FGSM_perturbed/FaceSwap/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/FGSM_perturbed/FaceSwap/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/FGSM_perturbed/FaceSwap/c23/test', exist_ok=True)

os.makedirs('/mnt/xjc/images/FGSM_perturbed/NeuralTextures/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/FGSM_perturbed/NeuralTextures/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/FGSM_perturbed/NeuralTextures/c23/test', exist_ok=True)
######################################################################

########################## PGD5 #################################
os.makedirs('/mnt/xjc/images/PGD5_perturbed/Deepfakes/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD5_perturbed/Deepfakes/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD5_perturbed/Deepfakes/c23/test', exist_ok=True)

os.makedirs('/mnt/xjc/images/PGD5_perturbed/Face2Face/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD5_perturbed/Face2Face/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD5_perturbed/Face2Face/c23/test', exist_ok=True)

os.makedirs('/mnt/xjc/images/PGD5_perturbed/FaceSwap/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD5_perturbed/FaceSwap/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD5_perturbed/FaceSwap/c23/test', exist_ok=True)

os.makedirs('/mnt/xjc/images/PGD5_perturbed/NeuralTextures/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD5_perturbed/NeuralTextures/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD5_perturbed/NeuralTextures/c23/test', exist_ok=True)
######################################################################

########################## PGD10 #################################
os.makedirs('/mnt/xjc/images/PGD10_perturbed/Deepfakes/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD10_perturbed/Deepfakes/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD10_perturbed/Deepfakes/c23/test', exist_ok=True)

os.makedirs('/mnt/xjc/images/PGD10_perturbed/Face2Face/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD10_perturbed/Face2Face/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD10_perturbed/Face2Face/c23/test', exist_ok=True)

os.makedirs('/mnt/xjc/images/PGD10_perturbed/FaceSwap/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD10_perturbed/FaceSwap/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD10_perturbed/FaceSwap/c23/test', exist_ok=True)

os.makedirs('/mnt/xjc/images/PGD10_perturbed/NeuralTextures/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD10_perturbed/NeuralTextures/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD10_perturbed/NeuralTextures/c23/test', exist_ok=True)
######################################################################

########################## PGD20 #################################
os.makedirs('/mnt/xjc/images/PGD20_perturbed/Deepfakes/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD20_perturbed/Deepfakes/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD20_perturbed/Deepfakes/c23/test', exist_ok=True)

os.makedirs('/mnt/xjc/images/PGD20_perturbed/Face2Face/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD20_perturbed/Face2Face/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD20_perturbed/Face2Face/c23/test', exist_ok=True)

os.makedirs('/mnt/xjc/images/PGD20_perturbed/FaceSwap/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD20_perturbed/FaceSwap/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD20_perturbed/FaceSwap/c23/test', exist_ok=True)

os.makedirs('/mnt/xjc/images/PGD20_perturbed/NeuralTextures/c23/train', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD20_perturbed/NeuralTextures/c23/val', exist_ok=True)
os.makedirs('/mnt/xjc/images/PGD20_perturbed/NeuralTextures/c23/test', exist_ok=True)
######################################################################

original_train = []
original_val = []
original_test = []

manipulated_train = []
manipulated_val = []
manipulated_test = []

perturbed_train = []
perturbed_val = []
perturbed_test = []

# train
with open('train.json', 'r') as f:
  train = json.load(f)
  for pair in train:
    original_train.append(pair[0] + '_frame10.jpg')
    original_train.append(pair[1] + '_frame10.jpg')
    original_train.append(pair[0] + '_frame20.jpg')
    original_train.append(pair[1] + '_frame20.jpg')
    original_train.append(pair[0] + '_frame30.jpg')
    original_train.append(pair[1] + '_frame30.jpg')
    original_train.append(pair[0] + '_frame40.jpg')
    original_train.append(pair[1] + '_frame40.jpg')

    manipulated_train.append(pair[0] + '_' + pair[1] + '_frame10.jpg')
    manipulated_train.append(pair[1] + '_' + pair[0] + '_frame10.jpg')

    perturbed_train.append(pair[0] + '_' + pair[1] + '.jpg')
    perturbed_train.append(pair[1] + '_' + pair[0] + '.jpg')
    

print('Moving the original train...')
for img_name in original_train:
  src = '/mnt/xjc/images/original/c23/' + img_name
  dst = '/mnt/xjc/images/original/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Moving the manipulated train...')
for img_name in manipulated_train:
  src = '/mnt/xjc/images/manipulated/Deepfakes/c23/' + img_name
  dst = '/mnt/xjc/images/manipulated/Deepfakes/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/manipulated/Face2Face/c23/' + img_name
  dst = '/mnt/xjc/images/manipulated/Face2Face/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/manipulated/FaceSwap/c23/' + img_name
  dst = '/mnt/xjc/images/manipulated/FaceSwap/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/manipulated/NeuralTextures/c23/' + img_name
  dst = '/mnt/xjc/images/manipulated/NeuralTextures/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Moving the perturbed train...')
for img_name in perturbed_train:
  src = '/mnt/xjc/images/FGSM_perturbed/Deepfakes/c23/' + img_name
  dst = '/mnt/xjc/images/FGSM_perturbed/Deepfakes/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/FGSM_perturbed/Face2Face/c23/' + img_name
  dst = '/mnt/xjc/images/FGSM_perturbed/Face2Face/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/FGSM_perturbed/FaceSwap/c23/' + img_name
  dst = '/mnt/xjc/images/FGSM_perturbed/FaceSwap/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/FGSM_perturbed/NeuralTextures/c23/' + img_name
  dst = '/mnt/xjc/images/FGSM_perturbed/NeuralTextures/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Moving the PGD5_perturbed train...')
for img_name in perturbed_train:
  src = '/mnt/xjc/images/PGD5_perturbed/Deepfakes/c23/' + img_name
  dst = '/mnt/xjc/images/PGD5_perturbed/Deepfakes/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD5_perturbed/Face2Face/c23/' + img_name
  dst = '/mnt/xjc/images/PGD5_perturbed/Face2Face/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD5_perturbed/FaceSwap/c23/' + img_name
  dst = '/mnt/xjc/images/PGD5_perturbed/FaceSwap/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD5_perturbed/NeuralTextures/c23/' + img_name
  dst = '/mnt/xjc/images/PGD5_perturbed/NeuralTextures/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Moving the PGD10_perturbed train...')
for img_name in perturbed_train:
  src = '/mnt/xjc/images/PGD10_perturbed/Deepfakes/c23/' + img_name
  dst = '/mnt/xjc/images/PGD10_perturbed/Deepfakes/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD10_perturbed/Face2Face/c23/' + img_name
  dst = '/mnt/xjc/images/PGD10_perturbed/Face2Face/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD10_perturbed/FaceSwap/c23/' + img_name
  dst = '/mnt/xjc/images/PGD10_perturbed/FaceSwap/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD10_perturbed/NeuralTextures/c23/' + img_name
  dst = '/mnt/xjc/images/PGD10_perturbed/NeuralTextures/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Moving the PGD20_perturbed train...')
for img_name in perturbed_train:
  src = '/mnt/xjc/images/PGD20_perturbed/Deepfakes/c23/' + img_name
  dst = '/mnt/xjc/images/PGD20_perturbed/Deepfakes/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD20_perturbed/Face2Face/c23/' + img_name
  dst = '/mnt/xjc/images/PGD20_perturbed/Face2Face/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD20_perturbed/FaceSwap/c23/' + img_name
  dst = '/mnt/xjc/images/PGD20_perturbed/FaceSwap/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD20_perturbed/NeuralTextures/c23/' + img_name
  dst = '/mnt/xjc/images/PGD20_perturbed/NeuralTextures/c23/train'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Train image split ends.')

# val
with open('val.json', 'r') as f:
  val = json.load(f)
  for pair in val:
    original_val.append(pair[0] + '_frame10.jpg')
    original_val.append(pair[1] + '_frame10.jpg')
    original_val.append(pair[0] + '_frame20.jpg')
    original_val.append(pair[1] + '_frame20.jpg')
    original_val.append(pair[0] + '_frame30.jpg')
    original_val.append(pair[1] + '_frame30.jpg')
    original_val.append(pair[0] + '_frame40.jpg')
    original_val.append(pair[1] + '_frame40.jpg')

    manipulated_val.append(pair[0] + '_' + pair[1] + '_frame10.jpg')
    manipulated_val.append(pair[1] + '_' + pair[0] + '_frame10.jpg')

    perturbed_val.append(pair[0] + '_' + pair[1] + '.jpg')
    perturbed_val.append(pair[1] + '_' + pair[0] + '.jpg')


print('Moving the original val...')
for img_name in original_val:
  src = '/mnt/xjc/images/original/c23/' + img_name
  dst = '/mnt/xjc/images/original/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Moving the manipulated val...')
for img_name in manipulated_val:
  src = '/mnt/xjc/images/manipulated/Deepfakes/c23/' + img_name
  dst = '/mnt/xjc/images/manipulated/Deepfakes/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/manipulated/Face2Face/c23/' + img_name
  dst = '/mnt/xjc/images/manipulated/Face2Face/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/manipulated/FaceSwap/c23/' + img_name
  dst = '/mnt/xjc/images/manipulated/FaceSwap/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/manipulated/NeuralTextures/c23/' + img_name
  dst = '/mnt/xjc/images/manipulated/NeuralTextures/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Moving the perturbed val...')
for img_name in perturbed_val:
  src = '/mnt/xjc/images/FGSM_perturbed/Deepfakes/c23/' + img_name
  dst = '/mnt/xjc/images/FGSM_perturbed/Deepfakes/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/FGSM_perturbed/Face2Face/c23/' + img_name
  dst = '/mnt/xjc/images/FGSM_perturbed/Face2Face/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/FGSM_perturbed/FaceSwap/c23/' + img_name
  dst = '/mnt/xjc/images/FGSM_perturbed/FaceSwap/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/FGSM_perturbed/NeuralTextures/c23/' + img_name
  dst = '/mnt/xjc/images/FGSM_perturbed/NeuralTextures/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Moving the PGD5_perturbed val...')
for img_name in perturbed_val:
  src = '/mnt/xjc/images/PGD5_perturbed/Deepfakes/c23/' + img_name
  dst = '/mnt/xjc/images/PGD5_perturbed/Deepfakes/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD5_perturbed/Face2Face/c23/' + img_name
  dst = '/mnt/xjc/images/PGD5_perturbed/Face2Face/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD5_perturbed/FaceSwap/c23/' + img_name
  dst = '/mnt/xjc/images/PGD5_perturbed/FaceSwap/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD5_perturbed/NeuralTextures/c23/' + img_name
  dst = '/mnt/xjc/images/PGD5_perturbed/NeuralTextures/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Moving the PGD10_perturbed val...')
for img_name in perturbed_val:
  src = '/mnt/xjc/images/PGD10_perturbed/Deepfakes/c23/' + img_name
  dst = '/mnt/xjc/images/PGD10_perturbed/Deepfakes/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD10_perturbed/Face2Face/c23/' + img_name
  dst = '/mnt/xjc/images/PGD10_perturbed/Face2Face/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD10_perturbed/FaceSwap/c23/' + img_name
  dst = '/mnt/xjc/images/PGD10_perturbed/FaceSwap/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD10_perturbed/NeuralTextures/c23/' + img_name
  dst = '/mnt/xjc/images/PGD10_perturbed/NeuralTextures/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Moving the PGD20_perturbed val...')
for img_name in perturbed_val:
  src = '/mnt/xjc/images/PGD20_perturbed/Deepfakes/c23/' + img_name
  dst = '/mnt/xjc/images/PGD20_perturbed/Deepfakes/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD20_perturbed/Face2Face/c23/' + img_name
  dst = '/mnt/xjc/images/PGD20_perturbed/Face2Face/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD20_perturbed/FaceSwap/c23/' + img_name
  dst = '/mnt/xjc/images/PGD20_perturbed/FaceSwap/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD20_perturbed/NeuralTextures/c23/' + img_name
  dst = '/mnt/xjc/images/PGD20_perturbed/NeuralTextures/c23/val'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Val image split ends.')

# test
with open('test.json', 'r') as f:
  test = json.load(f)
  for pair in test:
    original_test.append(pair[0] + '_frame10.jpg')
    original_test.append(pair[1] + '_frame10.jpg')
    original_test.append(pair[0] + '_frame20.jpg')
    original_test.append(pair[1] + '_frame20.jpg')
    original_test.append(pair[0] + '_frame30.jpg')
    original_test.append(pair[1] + '_frame30.jpg')
    original_test.append(pair[0] + '_frame40.jpg')
    original_test.append(pair[1] + '_frame40.jpg')

    manipulated_test.append(pair[0] + '_' + pair[1] + '_frame10.jpg')
    manipulated_test.append(pair[1] + '_' + pair[0] + '_frame10.jpg')

    perturbed_test.append(pair[0] + '_' + pair[1] + '.jpg')
    perturbed_test.append(pair[1] + '_' + pair[0] + '.jpg')

print('Moving the original test...')
for img_name in original_test:
  src = '/mnt/xjc/images/original/c23/' + img_name
  dst = '/mnt/xjc/images/original/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Moving the manipulated test...')
for img_name in manipulated_test:
  src = '/mnt/xjc/images/manipulated/Deepfakes/c23/' + img_name
  dst = '/mnt/xjc/images/manipulated/Deepfakes/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/manipulated/Face2Face/c23/' + img_name
  dst = '/mnt/xjc/images/manipulated/Face2Face/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/manipulated/FaceSwap/c23/' + img_name
  dst = '/mnt/xjc/images/manipulated/FaceSwap/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/manipulated/NeuralTextures/c23/' + img_name
  dst = '/mnt/xjc/images/manipulated/NeuralTextures/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Moving the perturbed test...')
for img_name in perturbed_test:
  src = '/mnt/xjc/images/FGSM_perturbed/Deepfakes/c23/' + img_name
  dst = '/mnt/xjc/images/FGSM_perturbed/Deepfakes/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/FGSM_perturbed/Face2Face/c23/' + img_name
  dst = '/mnt/xjc/images/FGSM_perturbed/Face2Face/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/FGSM_perturbed/FaceSwap/c23/' + img_name
  dst = '/mnt/xjc/images/FGSM_perturbed/FaceSwap/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/FGSM_perturbed/NeuralTextures/c23/' + img_name
  dst = '/mnt/xjc/images/FGSM_perturbed/NeuralTextures/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Moving the PGD5_perturbed test...')
for img_name in perturbed_test:
  src = '/mnt/xjc/images/PGD5_perturbed/Deepfakes/c23/' + img_name
  dst = '/mnt/xjc/images/PGD5_perturbed/Deepfakes/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD5_perturbed/Face2Face/c23/' + img_name
  dst = '/mnt/xjc/images/PGD5_perturbed/Face2Face/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD5_perturbed/FaceSwap/c23/' + img_name
  dst = '/mnt/xjc/images/PGD5_perturbed/FaceSwap/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD5_perturbed/NeuralTextures/c23/' + img_name
  dst = '/mnt/xjc/images/PGD5_perturbed/NeuralTextures/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Moving the PGD10_perturbed test...')
for img_name in perturbed_test:
  src = '/mnt/xjc/images/PGD10_perturbed/Deepfakes/c23/' + img_name
  dst = '/mnt/xjc/images/PGD10_perturbed/Deepfakes/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD10_perturbed/Face2Face/c23/' + img_name
  dst = '/mnt/xjc/images/PGD10_perturbed/Face2Face/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD10_perturbed/FaceSwap/c23/' + img_name
  dst = '/mnt/xjc/images/PGD10_perturbed/FaceSwap/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD10_perturbed/NeuralTextures/c23/' + img_name
  dst = '/mnt/xjc/images/PGD10_perturbed/NeuralTextures/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Moving the PGD20_perturbed test...')
for img_name in perturbed_test:
  src = '/mnt/xjc/images/PGD20_perturbed/Deepfakes/c23/' + img_name
  dst = '/mnt/xjc/images/PGD20_perturbed/Deepfakes/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD20_perturbed/Face2Face/c23/' + img_name
  dst = '/mnt/xjc/images/PGD20_perturbed/Face2Face/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD20_perturbed/FaceSwap/c23/' + img_name
  dst = '/mnt/xjc/images/PGD20_perturbed/FaceSwap/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

  src = '/mnt/xjc/images/PGD20_perturbed/NeuralTextures/c23/' + img_name
  dst = '/mnt/xjc/images/PGD20_perturbed/NeuralTextures/c23/test'
  try:
    shutil.move(src, dst)
  except:
    pass

print('Test image split ends.')