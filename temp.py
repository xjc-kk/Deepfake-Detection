import os
import shutil

train_original_path = '/mnt/xjc/images/original/c23/train'
val_original_path = '/mnt/xjc/images/original/c23/val'

global_path = '/mnt/xjc/images/PGD20_perturbed'
face_path = '/mnt/xjc/images/PGD20_regional_perturbed'
region_path = '/mnt/xjc/images/PGD20_smaller_region'

new_train_original_path = '/mnt/xjc/images/DirectMixTraining/train_original'
new_train_global_path = '/mnt/xjc/images/DirectMixTraining/train_global'
new_train_face_path = '/mnt/xjc/images/DirectMixTraining/train_face'
new_train_region_path = '/mnt/xjc/images/DirectMixTraining/train_region'

new_val_original_path = '/mnt/xjc/images/DirectMixTraining/val_original'
new_val_global_path = '/mnt/xjc/images/DirectMixTraining/val_global'
new_val_face_path = '/mnt/xjc/images/DirectMixTraining/val_face'
new_val_region_path = '/mnt/xjc/images/DirectMixTraining/val_region'

print('Moving Original Data...')
for img_name in os.listdir(train_original_path):
    src = os.path.join(train_original_path, img_name)
    dst = os.path.join(new_train_original_path, img_name)
    try:
        shutil.copyfile(src, dst)
    except:
        pass

for img_name in os.listdir(val_original_path):
    src = os.path.join(val_original_path, img_name)
    dst = os.path.join(new_val_original_path, img_name)
    try:
        shutil.copyfile(src, dst)
    except:
        pass

fake_list = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
print('Moving Global Data...')
for fake_type in fake_list:
    train_path = os.path.join(global_path, fake_type, 'c23/train')
    val_path = os.path.join(global_path, fake_type, 'c23/val')

    for img_name in os.listdir(train_path):
        src = os.path.join(train_path, img_name)
        dst = os.path.join(new_train_global_path, fake_type + '_' + img_name)
        try:
            shutil.copyfile(src, dst)
        except:
            pass

    for img_name in os.listdir(val_path):
        src = os.path.join(val_path, img_name)
        dst = os.path.join(new_val_global_path, fake_type + '_' + img_name)
        try:
            shutil.copyfile(src, dst)
        except:
            pass

print('Moving Face Data...')
for fake_type in fake_list:
    train_path = os.path.join(face_path, fake_type, 'c23/train')
    val_path = os.path.join(face_path, fake_type, 'c23/val')

    for img_name in os.listdir(train_path):
        src = os.path.join(train_path, img_name)
        dst = os.path.join(new_train_face_path, fake_type + '_' + img_name)
        try:
            shutil.copyfile(src, dst)
        except:
            pass

    for img_name in os.listdir(val_path):
        src = os.path.join(val_path, img_name)
        dst = os.path.join(new_val_face_path, fake_type + '_' + img_name)
        try:
            shutil.copyfile(src, dst)
        except:
            pass

print('Moving Region Data...')
for fake_type in fake_list:
    train_path = os.path.join(region_path, fake_type, 'c23/train')
    val_path = os.path.join(region_path, fake_type, 'c23/val')

    for img_name in os.listdir(train_path):
        src = os.path.join(train_path, img_name)
        dst = os.path.join(new_train_region_path, fake_type + '_' + img_name)
        try:
            shutil.copyfile(src, dst)
        except:
            pass

    for img_name in os.listdir(val_path):
        src = os.path.join(val_path, img_name)
        dst = os.path.join(new_val_region_path, fake_type + '_' + img_name)
        try:
            shutil.copyfile(src, dst)
        except:
            pass