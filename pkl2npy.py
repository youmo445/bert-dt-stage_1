import pickle
import numpy as np
import os
import re

# 定义读取pkl文件的函数
def read_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# 定义将pkl数据转换为npy格式的函数
def convert_to_npy(input_paths, output_path):
    merged_data = []
    trajectory_counts = []
    file_goals = []  # 用于存储每个文件对应的goal值
    
    for input_path in input_paths:
        # 提取文件名中的数字作为goal值
        file_name = os.path.basename(input_path)
        match = re.search(r'\d+', file_name)
        if match:
            goal_value = int(match.group())
        else:
            print(f"Skipping {input_path}: No digit found in filename.")
            continue
        
        # 读取pkl文件
        data = read_pkl(input_path)
        
        # 检查数据是否为列表且每个元素是字典
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            merged_data.extend(data)
            trajectory_counts.append(len(data))
            file_goals.append(goal_value)  # 记录该文件的goal值
        else:
            print(f"Skipping {input_path}: The data is not a list of dictionaries.")
    
    # 如果没有有效数据，则退出
    if not merged_data:
        print("No valid data to process.")
        return
    
    # 提取所有字段名
    keys = merged_data[0].keys()
    
    # 初始化一个字典来存储合并后的数据
    combined_data = {key: [] for key in keys}
    
    # 遍历每个字典，合并数据
    for item in merged_data:
        for key in keys:
            combined_data[key].append(item[key])
    
    # 将合并后的数据转换为NumPy数组
    for key in keys:
        combined_data[key] = np.array(combined_data[key])
    
    # 添加自定义字段
    num_trajectories = len(merged_data)
    if num_trajectories == 0:
        print("No trajectories found.")
        return
    
    trajectory_length = merged_data[0]['observations'].shape[0]
    
    # 创建 valid 字段
    valid = np.ones((num_trajectories, trajectory_length, 1), dtype=np.float32)
    combined_data['valid'] = valid
    
    # 创建 goal 字段
    goal = np.zeros((num_trajectories, 1), dtype=np.float32)
    start_idx = 0
    for i, count in enumerate(trajectory_counts):
        end_idx = start_idx + count
        goal[start_idx:end_idx] = file_goals[i]  # 使用文件名中的数字作为goal值
        start_idx = end_idx
    
    combined_data['goals'] = goal
    
    # 创建最终的字典结构
    final_data = {'fields': combined_data}
    
    # 保存为.npy文件
    output_directory = os.path.dirname(output_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    np.save(output_path, final_data)
    print(f"Data successfully saved to {output_path}")

# 主程序
if __name__ == "__main__":
    # 输入文件路径
    input_file_paths = [
        'data/cheetah_vel/cheetah_vel-0-expert.pkl',
        'data/cheetah_vel/cheetah_vel-6-expert.pkl',
        'data/cheetah_vel/cheetah_vel-32-expert.pkl',
        'data/cheetah_vel/cheetah_vel-33-expert.pkl',
        'data/cheetah_vel/cheetah_vel-35-expert.pkl',
        'data/cheetah_vel/cheetah_vel-36-expert.pkl'
    ]
    output_file_path = 'dataset/merged_cheetah_vel.npy'
    
    # 转换文件
    convert_to_npy(input_file_paths, output_file_path)