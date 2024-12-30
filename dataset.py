import os
import pandas as pd


folder_path = "./animals"
image_extensions = '.jpg'
data = []

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if os.path.splitext(file)[1].lower() == image_extensions:
            filepath = os.path.join(root, file)
            data.append({'filepath': filepath, 'label': 0})

folder_path = "./otter"

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if os.path.splitext(file)[1].lower() == image_extensions:
            filepath = os.path.join(root, file)
            data.append({'filepath': filepath, 'label': 1})

df = pd.DataFrame(data)
df.to_csv('image_data.csv')