'''
Mix PGD1 to PGD5 data into PGD1-5 dataset.
'''
import os
import shutil

fake_list = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
for fake_type in fake_list:
    train_path = os.path.join('/mnt/xjc/images/PGD1-5_regional_perturbed', fake_type, 'c23/train')
    val_path = os.path.join('/mnt/xjc/images/PGD1-5_regional_perturbed', fake_type, 'c23/val')
    test_path = os.path.join('/mnt/xjc/images/PGD1-5_regional_perturbed', fake_type, 'c23/test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

src_list = ['PGD1', 'PGD2', 'PGD3', 'PGD4', 'PGD5']
for iter_time in src_list:
    src_path = '/mnt/xjc/images/' + iter_time + '_regional_perturbed'
    fake_list = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    for fake_type in fake_list:
        # '/mnt/xjc/images/PGD1_perturbed/Deepfakes/c23'
        fake_path = os.path.join(src_path, fake_type, 'c23')
        data_list = ['train', 'val', 'test']
        for data_type in data_list:
            # '/mnt/xjc/images/PGD1_perturbed/Deepfakes/c23/train'
            path = os.path.join(fake_path, data_type)
            for img_name in os.listdir(path):
                dst_img_name = iter_time + '_' + img_name
                src = os.path.join(path, img_name)
                dst = os.path.join('/mnt/xjc/images/PGD1-5_regional_perturbed', fake_type, 'c23', data_type, dst_img_name)
                try:
                    shutil.copyfile(src, dst)
                except:
                    pass