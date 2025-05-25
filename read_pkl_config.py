import os
import pickle
import numpy as np

# 定义读取pkl文件的函数
def read_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# 定义打印字典中所有键及其形状的函数
def print_dict_keys_and_shapes(data, file_name):
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        print(f"\nFile: {file_name}")
        print("Data is a list of dictionaries. Printing keys and their shapes:")
        for i, item in enumerate(data):
            #print(f"\nDictionary {i + 1}:")
            for key, value in item.items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    shape = np.shape(value) if isinstance(value, np.ndarray) else len(value)
                    print(f"  Key: {key}, Shape: {shape}")
                else:
                    print(f"  Key: {key}, Value type: {type(value).__name__}, Value: {value}")
    else:
        print(f"\nFile: {file_name}")
        print("The data is not a list of dictionaries. Please check the data structure.")

# 主程序
if __name__ == "__main__":
    # 替换为你的config文件夹路径
    folder_path = 'config/ant_dir'

    # 获取文件夹下所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pkl'):  # 只处理pkl文件
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")
            try:
                data = read_pkl(file_path)
                print_dict_keys_and_shapes(data, file_name)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")