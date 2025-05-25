import pickle
import numpy as np
# 定义读取pkl文件的函数
def read_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# 定义打印字典中所有键及其形状的函数
def print_dict_keys_and_shapes(data):
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        print("Data is a list of dictionaries. Printing keys and their shapes:")
        for i, item in enumerate(data):
            print(f"\nDictionary {i + 1}:")
            for key, value in item.items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    shape = np.shape(value) if isinstance(value, np.ndarray) else len(value)
                    print(f"  Key: {key}, Shape: {shape}")
                else:
                    print(f"  Key: {key}, Value type: {type(value).__name__}, Value: {value}")
    else:
        print("The data is not a list of dictionaries. Please check the data structure.")

# 主程序
if __name__ == "__main__":
    # 替换为你的pkl文件路径
    pkl_file_path = 'preprocessed_vectors.pkl'
    #pkl_file_path = 'data/ML1-pick-place-v2/ML1-pick-place-v2-0-expert.pkl'
    data = read_pkl(pkl_file_path)
    print_dict_keys_and_shapes(data)