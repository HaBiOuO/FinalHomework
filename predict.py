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
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3)
        self.conv7 = nn.Conv2d(1024, 512, kernel_size=3)
        self.conv8 = nn.Conv2d(512,256,kernel_size=3)
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        x = F.relu(self.conv7(x))
        x = self.pool(F.relu(self.conv8(x)))
        x = self.flatten(x)  
        x = self.fc1(x)
        x = self.fc3(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('model.pth', map_location=device)
model.eval()
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

labels = ["otter", "other"]

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    predicted_label = labels[predicted.item()]
    return predicted_label



folder_path = "your test folder"
image_extensions = '.jpg'
datas = []
total = 0

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if os.path.splitext(file)[1].lower() == image_extensions:
            filepath = os.path.join(root, file)
            datas.append({'filepath': filepath, 'label': 1})

correct = 0
for data in datas:
    result = predict_image(data['filepath'])
    if(result == 'other'):
        print(f"Not otter")
    else:
        correct +=1 
        print(f"The predicted label for the image is: {result}")
    total += 1

print(f'Accuracy:{correct/total}')