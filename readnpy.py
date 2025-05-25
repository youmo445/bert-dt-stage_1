import numpy as np
import os

# 定义文件路径
file_path = "dataset/cheetah_dir-0-expert.npy"
data = np.load(file_path, allow_pickle=True).item()
# 检查data的类型
if isinstance(data, dict):
    # 确保data是一个字典
    print("Data is a dictionary.")
    fields = data.get('fields')  # 使用.get()方法安全地访问键
    if fields is not None:
        # 如果fields存在，打印每个键的形状
        for key, value in fields.items():
            print(f"{key}: {value.shape}")
    else:
        print("Key 'fields' not found in the data.")
else:
    print(f"Data is not a dictionary. Type of data: {type(data)}")