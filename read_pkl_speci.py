import pickle
import numpy as np

# 定义读取pkl文件的函数
def read_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# 定义打印特定字典中指定键的值
def print_specific_dict_values(data, dict_index, keys_to_print):
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        if 0 <= dict_index < len(data):  # 检查索引是否有效
            selected_dict = data[dict_index]
            print(f"\nDictionary {dict_index + 1}:")
            for key in keys_to_print:
                if key in selected_dict:
                    value = selected_dict[key]
                    if isinstance(value, np.ndarray):
                        print(f"  Key: {key}, Shape: {value.shape}")
                        if key == 'terminals':  # 如果是 terminals，打印具体值
                            print(f"  Terminals Values: {value}")
                    else:
                        print(f"  Key: {key}, Values: {value}")
                else:
                    print(f"  Key: {key} not found in this dictionary.")
        else:
            print(f"Invalid dictionary index: {dict_index}. Index out of range.")
    else:
        print("The data is not a list of dictionaries. Please check the data structure.")

# 主程序
if __name__ == "__main__":
    # 替换为你的pkl文件路径
    pkl_file_path = 'data/cheetah_vel/cheetah_vel-0-expert.pkl'
    data = read_pkl(pkl_file_path)
    
    # 选择第999个字典（索引为998），并打印指定的键
    dict_index = 2  # 索引从0开始
    keys_to_print = ['actions', 'rewards', 'terminals']
    print_specific_dict_values(data, dict_index, keys_to_print)