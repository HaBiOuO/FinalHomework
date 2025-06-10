import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm 
import os

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'animals'
train_dataset = datasets.ImageFolder(os.path.join(data_dir), data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
num_classes = len(train_dataset.classes)

model = models.resnet101(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    running_corrects = 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
    for inputs, labels in loop:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        loop.set_postfix({
            'Loss': loss.item(),
            'Acc': (torch.sum(preds == labels.data).double() / inputs.size(0)).item()
        })
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    print(f"Epoch {epoch+1} Completed - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
print("訓練完成！")
torch.save(model.state_dict(), "finetuned.pth")
