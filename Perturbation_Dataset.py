import os
import json
import cv2
import torch
from PIL import Image
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
    image = transforms.Resize((height, width))(Image.fromarray(image))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image


src_path = '/mnt/xjc/cvpr_images'
dst_path = '/mnt/xjc/cvpr_images'

model_path = '/mnt/xjc/Deepfake-Detection/cvpr_models/'

frame_press_rate = ['FFppDFaceC0', 'FFppDFaceC23', 'FFppDFaceC40']
fake_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
data_types = ['train', 'val', 'test']

model = torch.load(model_path)
model = model.cuda()

# make dirs
for pr in frame_press_rate:
    for fake_type in fake_types:
        for data_type in data_types:
            os.makedirs(os.path.join(dst_path, pr, 'global_perturbed', fake_type, data_type), exist_ok=True)

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
        for data_type in data_types:
            fake_data_path = os.path.join(fake_path, data_type)
            for fake_name in os.listdir(fake_data_path):
                fake_video_path = os.path.join(fake_path, fake_name)
                all_frames = sorted(os.listdir(fake_video_path))
                frame_num = len(all_frames)

                for img_name in all_frames:
                    img_path = os.path.join(fake_video_path, img_name)
                    img = cv2.imread(img_path)
                    perturbed_img = PGD_attack(img, model, iters=20)
                    perturbed_imgdir_path = os.path.join(dst_path, pr, 'global_perturbed', fake_type, data_type, fake_name)
                    os.makedirs(perturbed_imgdir_path, exist_ok=True)
                    perturbed_img_path = os.path.join(perturbed_imgdir_path, img_name)
                    cv2.imwrite(perturbed_img_path, perturbed_img)