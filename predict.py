import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os

preprocess = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

image_path = 'dog.jpg'
input_image = Image.open(image_path).convert('RGB')
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet101(pretrained=False)
num_classes = 89  
model.fc = nn.Linear(model.fc.in_features, num_classes)


model.load_state_dict(torch.load("finetuned_googlenet.pth", map_location=device))
model.to(device)
model.eval()

class_names = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']

with torch.no_grad():
    input_batch = input_batch.to(device)
    output = model(input_batch)
    _, predicted = torch.max(output, 1)
    predicted_class = class_names[predicted.item()]


image_with_text = input_image.copy()
draw = ImageDraw.Draw(image_with_text)
font = ImageFont.load_default(size=50)

text = f"{predicted_class}"
text_position = (10, 10)
text_color = (255, 0, 0)
draw.text(text_position, text, fill=text_color, font=font)
plt.imshow(image_with_text)
image_with_text.save("output_with_label.jpg")  

