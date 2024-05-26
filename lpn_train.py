import torch
import torch.nn as nn
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from tqdm import tqdm


label_mapping = {"Latin": 0, "Korean": 1, "Japanese": 2, "Chinese": 3}
labels = {}

with open("/home/pirl/Desktop/OCR/FAST/data/testset/lpn/gts.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        image_name, label = line.strip().split('\t')
        labels[image_name] = label_mapping[label]

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


# 데이터 로딩 및 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet과 같은 일반적인 모델에 맞게 크기 조절
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 일반적인 정규화 값
])

# labels = {"tr_img_07219_2.jpg" : 0, "tr_img_05818_2.jpg" : 1 , "tr_img_03765_3.jpg" : 2 ,"tr_img_06264_8.jpg" : 3}  # 실제 이미지 파일 이름 및 라벨에 따라 수정
dataset = Language_dataset(img_dir="/home/pirl/Desktop/OCR/FAST/data/testset/lpn/imgs/", labels=labels, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# 4. 학습 설정
model = resnet18()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. 학습 루프
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, total=len(train_loader))
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        # print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        progress_bar.set_postfix(Loss=loss.item())

    
    
    # 간단한 학습 상태 출력
    accuracy = 100 * total_correct/total_samples
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

print("Training complete.")
torch.save(model.state_dict(), 'resnet18_model.pth')