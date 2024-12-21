import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import warnings
import os
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv4 = nn.Conv2d(256, 1024, kernel_size=3)
        self.conv5 = nn.Conv2d(1024, 512, kernel_size=3)
        self.conv6 = nn.Conv2d(512, 256, kernel_size=3)
        self.dropout = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(73728, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128,num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.flatten(x)  
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加載模型
model = torch.load('model.pth', map_location=device)
model.eval()
model = model.to(device)

# 定義圖片轉換（需與訓練時一致）
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

labels = ["other", "otter"]

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    predicted_label = labels[predicted.item()]
    return predicted_label



folder_path = "./valid"
image_extensions = '.jpg'
datas = []

# 遍歷資料夾及子資料夾
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if os.path.splitext(file)[1].lower() == image_extensions:
            # 完整檔案路徑
            filepath = os.path.join(root, file)
            # 標籤是資料夾名稱
            # 將資料加入列表
            datas.append({'filepath': filepath, 'label': 1})


for data in datas:
    result = predict_image(data['filepath'])
    if(result == 'other'):
        print("Not Otter!")
    else:
        print(f"The predicted label for the image is: {result}")