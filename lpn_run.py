import torch
import torch.nn as nn
import os
import numpy as np
import cv2
import pdb
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from tqdm import tqdm

from torchvision import models
from collections import defaultdict
import torch.nn.functional as F



# label_mapping = {"Latin": 0, "Korean": 1, "Japanese": 2, "Chinese": 3}
# labels = {}

# with open("/home/pirl/Desktop/OCR/FAST/data/MLT19/lpn/gts.txt", 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         image_name, label = line.strip().split('\t')
#         labels[image_name] = label_mapping[label]

class Language_dataset(Dataset):

    
    def __init__(self, img_dir, labels, transform=None):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")  # RGB 형태로 이미지를 열기

        label = self.labels[img_name]  # 라벨 가져오기

        if self.transform:
            image = self.transform(image)

        return image , label

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = F.softmax(x, dim = 1)
        return x

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

model = resnet18()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. 체크포인트 로드
model_path = '/home/pirl/Desktop/OCR/FAST/1024_resnet18_weight.pth'
model.load_state_dict(torch.load(model_path, map_location=device))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_language(image_data, model=model, transform=transform):
    pdb.set_trace()
    if isinstance(image_data, torch.Tensor):
        if image_data.size == 0 :
            raise ValueError("The provided tensor is empty")
        image = image_data
    elif isinstance(image_data, np.ndarray):
        if image_data.size == 0 :
            raise ValueError("The provided numpy array is empty")
        
        if image_data.shape[2] != 3 :
            raise ValueError(f"Expected a 3-channel image, but got {image_data.shape[2]} channels.")

        image = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
    else :
        image = Image.open(image_data).convert("RGB")
    
    if len(image.shape) == 3:
        image = transform(image).unsqueeze(0)  # 차원 추가
    else:
        image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)

    # 이미지를 모델과 동일한 디바이스로 이동
    device = next(model.parameters()).device
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    index_to_label = {0: "Latin", 1: "Korean", 2: "Japanese", 3: "Chinese"}
    for i, pred in enumerate(predicted):

    return index_to_label[predicted.item()]


def predict_languages_in_folder(folder_path, model=model, transform=transform):
    
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
    predicted_languages = {}

    for filename in tqdm(image_files, desc="Predicting languages"):
        image_path = os.path.join(folder_path, filename)
        predicted_language = predict_language(model, image_path, transform)
        predicted_languages[filename] = predicted_language

    return predicted_languages

