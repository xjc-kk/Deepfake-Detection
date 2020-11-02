import os
import json
import shutil

src_path = '/mnt/FFpp'
dst_path = '/mnt/xjc/cvpr_images'

frame_press_rate = ['FFppDFaceC0', 'FFppDFaceC23', 'FFppDFaceC40']
fake_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
data_types = ['train', 'val', 'test']

# make dirs
for pr in frame_press_rate:
    for fake_type in fake_types:
        for data_type in data_types:
            os.makedirs(os.path.join(dst_path, pr, 'manipulated' fake_type, data_type), exist_ok=True)

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

# sample frames into cvpr_images folder
for pr in frame_press_rate:
    print('Processing press rate ' + pr)
    frame_path = os.path.join(src_path, pr, 'manipulated')
    for fake_type in fake_types:
        print('Processing ' + fake_type)
        fake_path = os.path.join(frame_path, fake_type)
        for fake_name in os.listdir(fake_path):
            fake_video_path = os.path.join(fake_path, fake_name)
            all_frames = sorted(os.listdir(fake_video_path))
            frame_num = len(all_frames)

            data_type, sample_num = None, None
            if fake_name in train_list:
                data_type = 'train'
                sample_num = 270
            elif fake_name in val_list:
                data_type = 'val'
                sample_num = 270
            else:
                data_type = 'test'
                sample_num = 100

            interval = int(frame_num / sample_num)
            sample_index = [i*interval for i in range(sample_num)]
            for ind in sample_index:
                img_name = all_frames[ind]
                img_path = os.path.join(fake_video_path, img_name)
                new_img_folder_path = os.path.join(dst_path, pr, 'manipulated', fake_type, data_type, fake_name)
                os.makedirs(new_img_folder_path, exist_ok=True)
                new_img_path = os.path.join(new_img_folder_path, img_name)
                try:
                    shutil.copyfile(img_path, new_img_path)
                except:
                    pass