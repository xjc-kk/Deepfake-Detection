'''
Splice images in PGD20_perturbed and original into patches.
Move images from PGD20_regional_perturbed and PGD20_smaller_region_patches.
'''
import os
import shutil
import cv2


new_train_original_path = '/mnt/xjc/images/MixTraining/train_original'
new_train_global_path = '/mnt/xjc/images/MixTraining/train_global'
new_train_face_path = '/mnt/xjc/images/MixTraining/train_face'
new_train_region_path = '/mnt/xjc/images/MixTraining/train_region'

new_val_original_path = '/mnt/xjc/images/MixTraining/val_original'
new_val_global_path = '/mnt/xjc/images/MixTraining/val_global'
new_val_face_path = '/mnt/xjc/images/MixTraining/val_face'
new_val_region_path = '/mnt/xjc/images/MixTraining/val_region'

fake_list = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

# Move images from PGD20_smaller_region_patches to new_train_region_path / new_val_region_path
print('Processing PGD20_smaller_region patches...')
for fake_type in fake_list:
    train_path = os.path.join('/mnt/xjc/images/PGD20_smaller_region_patches', fake_type, 'c23/train')
    for img_name in os.listdir(train_path):
        src = os.path.join(train_path, img_name)
        label, suffix = img_name.split('-')[0], img_name.split('-')[1]
        new_img_name = label + '-' + fake_type + '_' + suffix
        dst = os.path.join(new_train_region_path, new_img_name)
        try:
            shutil.copyfile(src, dst)
        except:
            pass

    val_path = os.path.join('/mnt/xjc/images/PGD20_smaller_region_patches', fake_type, 'c23/val')
    for img_name in os.listdir(val_path):
        src = os.path.join(val_path, img_name)
        label, suffix = img_name.split('-')[0], img_name.split('-')[1]
        new_img_name = label + '-' + fake_type + '_' + suffix
        dst = os.path.join(new_val_region_path, new_img_name)
        try:
            shutil.copyfile(src, dst)
        except:
            pass

print('Processing PGD20_regional_perturbed patches...')
for fake_type in fake_list:
    train_path = os.path.join('/mnt/xjc/images/PGD20_regional_perturbed_patches', fake_type, 'c23/train')
    for img_name in os.listdir(train_path):
        src = os.path.join(train_path, img_name)
        label, suffix = img_name.split('-')[0], img_name.split('-')[1]
        new_img_name = label + '-' + fake_type + '_' + suffix
        dst = os.path.join(new_train_face_path, new_img_name)
        try:
            shutil.copyfile(src, dst)
        except:
            pass

    val_path = os.path.join('/mnt/xjc/images/PGD20_regional_perturbed_patches', fake_type, 'c23/val')
    for img_name in os.listdir(val_path):
        src = os.path.join(val_path, img_name)
        label, suffix = img_name.split('-')[0], img_name.split('-')[1]
        new_img_name = label + '-' + fake_type + '_' + suffix
        dst = os.path.join(new_val_face_path, new_img_name)
        try:
            shutil.copyfile(src, dst)
        except:
            pass

# Splice images in original into patches
print('Processing original patches...')
original_train_path = '/mnt/xjc/images/original/c23/train'
for img_name in os.listdir(original_train_path):
    img_path = os.path.join(original_train_path, img_name)
    img = cv2.imread(img_path)

    h, w, c = img.shape
    patch_h, patch_w = int(h / 2), int(w / 2)
    patch1 = img[:patch_h, :patch_w, :]
    patch2 = img[:patch_h, patch_w:, :]
    patch3 = img[patch_h:, :patch_w, :]
    patch4 = img[patch_h:, patch_w:, :]
    patches = [patch1, patch2, patch3, patch4]

    for i in range(4):
        cv2.imwrite(os.path.join(new_train_original_path, '0-' + str(i) + '_' + img_name), patches[i])

original_val_path = '/mnt/xjc/images/original/c23/val'
for img_name in os.listdir(original_val_path):
    img_path = os.path.join(original_val_path, img_name)
    img = cv2.imread(img_path)

    h, w, c = img.shape
    patch_h, patch_w = int(h / 2), int(w / 2)
    patch1 = img[:patch_h, :patch_w, :]
    patch2 = img[:patch_h, patch_w:, :]
    patch3 = img[patch_h:, :patch_w, :]
    patch4 = img[patch_h:, patch_w:, :]
    patches = [patch1, patch2, patch3, patch4]

    for i in range(4):
        cv2.imwrite(os.path.join(new_val_original_path, '0-' + str(i) + '_' + img_name), patches[i])

# Splice images in PGD20_perturbed into patches
print('Processing PGD20_perturbed patches...')
global_path = '/mnt/xjc/images/PGD20_perturbed'
for fake_type in fake_list:
    train_path = os.path.join(global_path, fake_type, 'c23/train')
    for img_name in os.listdir(train_path):
        img_path = os.path.join(train_path, img_name)
        img = cv2.imread(img_path)

        h, w, c = img.shape
        patch_h, patch_w = int(h / 2), int(w / 2)
        patch1 = img[:patch_h, :patch_w, :]
        patch2 = img[:patch_h, patch_w:, :]
        patch3 = img[patch_h:, :patch_w, :]
        patch4 = img[patch_h:, patch_w:, :]
        patches = [patch1, patch2, patch3, patch4]

        for i in range(4):
            cv2.imwrite(os.path.join(new_train_global_path, '1-' + str(i) + '_' + fake_type + '_' + img_name), patches[i])

    val_path = os.path.join(global_path, fake_type, 'c23/val')
    for img_name in os.listdir(val_path):
        img_path = os.path.join(val_path, img_name)
        img = cv2.imread(img_path)

        h, w, c = img.shape
        patch_h, patch_w = int(h / 2), int(w / 2)
        patch1 = img[:patch_h, :patch_w, :]
        patch2 = img[:patch_h, patch_w:, :]
        patch3 = img[patch_h:, :patch_w, :]
        patch4 = img[patch_h:, patch_w:, :]
        patches = [patch1, patch2, patch3, patch4]

        for i in range(4):
            cv2.imwrite(os.path.join(new_val_global_path, '1-' + str(i) + '_' + fake_type + '_' + img_name), patches[i])