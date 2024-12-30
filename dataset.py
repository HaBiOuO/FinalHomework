import os
import pandas as pd


folder_path = "./animals"
image_extensions = '.jpg'
data = []

# 遍歷資料夾及子資料夾
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if os.path.splitext(file)[1].lower() == image_extensions:
            # 完整檔案路徑
            filepath = os.path.join(root, file)
            # 標籤是資料夾名稱
            # 將資料加入列表
            data.append({'filepath': filepath, 'label': 0})

folder_path = "./otter"

# 遍歷資料夾及子資料夾
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if os.path.splitext(file)[1].lower() == image_extensions:
            # 完整檔案路徑
            filepath = os.path.join(root, file)
            # 標籤是資料夾名稱
            # 將資料加入列表
            data.append({'filepath': filepath, 'label': 1})

df = pd.DataFrame(data)
df.to_csv('image_data.csv')