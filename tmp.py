import os
import shutil

src = '/mnt/xjc/images/original/c23/test'
dst = '/mnt/xjc/images/DirectMixTraining/test_original'
for img_name in os.listdir(src):
    img_path = os.path.join(src, img_name)
    dst_path = os.path.join(dst, img_name)
    try:
        shutil.copyfile(img_path, dst_path)
    except:
        pass

src = '/mnt/xjc/images/PGD20_perturbed/Deepfakes/c23/test'
dst = '/mnt/xjc/images/DirectMixTraining/test_global_Deepfakes'
for img_name in os.listdir(src):
    img_path = os.path.join(src, img_name)
    dst_path = os.path.join(dst, img_name)
    try:
        shutil.copyfile(img_path, dst_path)
    except:
        pass

src = '/mnt/xjc/images/PGD20_perturbed/Face2Face/c23/test'
dst = '/mnt/xjc/images/DirectMixTraining/test_global_Face2Face'
for img_name in os.listdir(src):
    img_path = os.path.join(src, img_name)
    dst_path = os.path.join(dst, img_name)
    try:
        shutil.copyfile(img_path, dst_path)
    except:
        pass

src = '/mnt/xjc/images/PGD20_perturbed/FaceSwap/c23/test'
dst = '/mnt/xjc/images/DirectMixTraining/test_global_FaceSwap'
for img_name in os.listdir(src):
    img_path = os.path.join(src, img_name)
    dst_path = os.path.join(dst, img_name)
    try:
        shutil.copyfile(img_path, dst_path)
    except:
        pass

src = '/mnt/xjc/images/PGD20_perturbed/NeuralTextures/c23/test'
dst = '/mnt/xjc/images/DirectMixTraining/test_global_NeuralTextures'
for img_name in os.listdir(src):
    img_path = os.path.join(src, img_name)
    dst_path = os.path.join(dst, img_name)
    try:
        shutil.copyfile(img_path, dst_path)
    except:
        pass

src = '/mnt/xjc/images/PGD20_regional_perturbed/Deepfakes/c23/test'
dst = '/mnt/xjc/images/DirectMixTraining/test_face_Deepfakes'
for img_name in os.listdir(src):
    img_path = os.path.join(src, img_name)
    dst_path = os.path.join(dst, img_name)
    try:
        shutil.copyfile(img_path, dst_path)
    except:
        pass

src = '/mnt/xjc/images/PGD20_regional_perturbed/Face2Face/c23/test'
dst = '/mnt/xjc/images/DirectMixTraining/test_face_Face2Face'
for img_name in os.listdir(src):
    img_path = os.path.join(src, img_name)
    dst_path = os.path.join(dst, img_name)
    try:
        shutil.copyfile(img_path, dst_path)
    except:
        pass

src = '/mnt/xjc/images/PGD20_regional_perturbed/FaceSwap/c23/test'
dst = '/mnt/xjc/images/DirectMixTraining/test_face_FaceSwap'
for img_name in os.listdir(src):
    img_path = os.path.join(src, img_name)
    dst_path = os.path.join(dst, img_name)
    try:
        shutil.copyfile(img_path, dst_path)
    except:
        pass

src = '/mnt/xjc/images/PGD20_regional_perturbed/NeuralTextures/c23/test'
dst = '/mnt/xjc/images/DirectMixTraining/test_face_NeuralTextures'
for img_name in os.listdir(src):
    img_path = os.path.join(src, img_name)
    dst_path = os.path.join(dst, img_name)
    try:
        shutil.copyfile(img_path, dst_path)
    except:
        pass

src = '/mnt/xjc/images/PGD20_smaller_region/Deepfakes/c23/test'
dst = '/mnt/xjc/images/DirectMixTraining/test_region_Deepfakes'
for img_name in os.listdir(src):
    img_path = os.path.join(src, img_name)
    dst_path = os.path.join(dst, img_name)
    try:
        shutil.copyfile(img_path, dst_path)
    except:
        pass

src = '/mnt/xjc/images/PGD20_smaller_region/Face2Face/c23/test'
dst = '/mnt/xjc/images/DirectMixTraining/test_region_Face2Face'
for img_name in os.listdir(src):
    img_path = os.path.join(src, img_name)
    dst_path = os.path.join(dst, img_name)
    try:
        shutil.copyfile(img_path, dst_path)
    except:
        pass

src = '/mnt/xjc/images/PGD20_smaller_region/FaceSwap/c23/test'
dst = '/mnt/xjc/images/DirectMixTraining/test_region_FaceSwap'
for img_name in os.listdir(src):
    img_path = os.path.join(src, img_name)
    dst_path = os.path.join(dst, img_name)
    try:
        shutil.copyfile(img_path, dst_path)
    except:
        pass

src = '/mnt/xjc/images/PGD20_smaller_region/NeuralTextures/c23/test'
dst = '/mnt/xjc/images/DirectMixTraining/test_region_NeuralTextures'
for img_name in os.listdir(src):
    img_path = os.path.join(src, img_name)
    dst_path = os.path.join(dst, img_name)
    try:
        shutil.copyfile(img_path, dst_path)
    except:
        pass