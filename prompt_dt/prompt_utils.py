import numpy as np
import gym
import json, pickle, random, os, torch
from collections import namedtuple
from .prompt_evaluate_episodes import prompt_evaluate_episode, prompt_evaluate_episode_rtg

# for mujoco tasks
from mujoco_control_envs.mujoco_control_envs import HalfCheetahDirEnv, HalfCheetahVelEnv, AntDirEnv
# for jacopinpad
from jacopinpad.jacopinpad_gym import jacopinpad_multi
from context.model import RNNContextEncoder, RewardDecoder, StateDecoder, descriptionsEecoder
# for metaworld
import metaworld
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
import torch
from transformers import BertTokenizer, BertModel
import json
import  numpy as np
from key import id2key
from use_moco import use_moco
# 指定本地模型和分词器的路径
model_path = './models/bert-base-uncased'
""" constructing envs """

import os
import torch
from transformers import BertTokenizer, BertModel
import pickle

def preprocess_descriptions(desc_dir, output_path, model_path, device='cuda'):
    """
    预处理描述文件，将描述输入到 BERT 模型中，生成 768 维向量并保存到磁盘。
    
    Args:
        desc_dir (str): 描述文件所在目录。
        output_path (str): 保存预处理向量的路径。
        model_path (str): BERT 模型路径。
        device (str): 设备（默认 'cuda'）。
    """
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path).to(device)
    model.eval()

    all_vectors = {}  # 用于存储每个描述文件的向量

    for desc_file in os.listdir(desc_dir):
        if desc_file.endswith('.py'):  # 确保是描述文件
            file_path = os.path.join(desc_dir, desc_file)
            with open(file_path, 'r') as f:
                descriptions = f.readlines()

            # 对每个描述生成 BERT 向量
            inputs = tokenizer(descriptions, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                vectors = outputs.last_hidden_state[:, 0, :]  # [num_descriptions, 768]
                print('vectors.shape',vectors.shape)
            all_vectors[desc_file] = vectors.cpu().numpy()  # 保存为 numpy 数组

    # 保存所有向量到磁盘
    with open(output_path, 'wb') as f:
        pickle.dump(all_vectors, f)
    print(f"Preprocessed vectors saved to {output_path}")
    
def gen_env(env_name, config_save_path):
    print('enter-prompt-utils_gen_env')
    if 'cheetah_dir' in env_name:
        if '0' in env_name:
            env = HalfCheetahDirEnv([{'direction': 1}], include_goal = False)
        # 如果env_name中包含'0'，则选择创建一个方向为正向（direction: 1）的HalfCheetahDirEnv环境，
        # 并且不包含目标（include_goal=False）。
        elif '1' in env_name:
            env = HalfCheetahDirEnv([{'direction': -1}], include_goal = False)
        max_ep_len = 200
        env_targets = [1500]
        scale = 1000.
    elif 'cheetah_vel' in env_name:
        task_idx = int(env_name.split('-')[-1])
        task_paths = f"{config_save_path}/cheetah_vel/config_cheetah_vel_task{task_idx}.pkl"
        tasks = []
        with open(task_paths.format(task_idx), 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
        env = HalfCheetahVelEnv(tasks, include_goal = False)
        max_ep_len = 200
        env_targets = [0]
        scale = 500.
    elif 'ant_dir' in env_name:
        task_idx = int(env_name.split('-')[-1])
        task_paths = f"{config_save_path}/ant_dir/config_ant_dir_task{task_idx}.pkl"
        tasks = []
        with open(task_paths.format(task_idx), 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
        env = AntDirEnv(tasks, len(tasks), include_goal = False)
        max_ep_len = 200
        env_targets = [500]
        scale = 500.
    elif 'ML1-' in env_name: # metaworld ML1
        task_name = '-'.join(env_name.split('-')[1:-1])
        ml1 = metaworld.ML1(task_name, seed=1) # Construct the benchmark, sampling tasks, note: our example datasets also have seed=1.
        env = ml1.train_classes[task_name]()  # Create an environment with task
        task_idx = int(env_name.split('-')[-1])
        task = ml1.train_tasks[task_idx]
        env.set_task(task)  # Set task
        max_ep_len = 500 
        env_targets= [int(650)]
        scale = 650.
    else:
        raise NotImplementedError
    return env, max_ep_len, env_targets, scale


def get_env_list(env_name_list, config_save_path, device):#先读训练集，再读测试集
    print('enter-prompt-utils_get-env-list')
    info = {} # store all the attributes for each env
    env_list = []
    
    for env_name in env_name_list:# cheetah_dir-0,cheetah_dir-1
        #{'max_ep_len': 200, 'env_targets': [1500], 'scale': 1000.0, 'state_dim': 20, 'act_dim': 6, 'device': 'cuda'}
        info[env_name] = {}
        env, max_ep_len, env_targets, scale = gen_env(env_name=env_name, config_save_path=config_save_path)
        info[env_name]['max_ep_len'] = max_ep_len
        info[env_name]['env_targets'] = env_targets
        info[env_name]['scale'] = scale
        info[env_name]['state_dim'] = env.observation_space.shape[0]
        info[env_name]['act_dim'] = env.action_space.shape[0] 
        info[env_name]['device'] = device
        env_list.append(env)
    return info, env_list


""" prompts """

def flatten_prompt(prompt, batch_size):
    print('enter-prompt-utils_flatten_prompt')
    p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask = prompt
    p_s = p_s.reshape((batch_size, -1, p_s.shape[-1]))
    p_a = p_a.reshape((batch_size, -1, p_a.shape[-1]))
    p_r = p_r.reshape((batch_size, -1, p_r.shape[-1]))
    p_d = p_d.reshape((batch_size, -1))
    p_rtg = p_rtg[:,:-1,:]
    p_rtg = p_rtg.reshape((batch_size, -1, p_rtg.shape[-1]))
    p_timesteps = p_timesteps.reshape((batch_size, -1))
    p_mask = p_mask.reshape((batch_size, -1))
    return p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask

def flatten_descriptions(descriptions, batch_size):
    print('enter-prompt-utils_flatten_descriptions')
    descriptions = descriptions.reshape((batch_size, -1, descriptions.shape[-1]))#(1,n,728)/(16,n,728)
    print('descriptions.shape',descriptions.shape)
    return descriptions

def get_prompt(prompt_trajectories, info, variant):
    print('enter-prompt-utils_get_prompt')
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_episodes, max_len = variant['prompt_episode'], variant['prompt_length']

    def fn(sample_size=1):
        # random sample prompts with fixed length (prompt-length) in num episodes (prompt-episode)
        #print('sample_size:',sample_size)#第一次是1，后面是16
        batch_inds = np.random.choice(
            np.arange(len(prompt_trajectories)),#5
            size=int(num_episodes*sample_size),#第一次是1*1，后面是1*16这两个变量是什么东西
            replace=True,
            # p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(int(num_episodes*sample_size)):
            if variant["stochastic_prompt"]:
                traj = prompt_trajectories[int(batch_inds[i])] # random select traj
            else:
                traj = prompt_trajectories[int(sorted_inds[-i])] # select the best traj with highest rewards
                # traj = prompt_trajectories[i]
            si = max(0, traj['rewards'].shape[0] - max_len -1) # select the last traj with length max_len，第一个traj.shape[0]--->200,max_len--->5

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)#(1,5,20)
        #print('s.shape',s.shape)
        #input()
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)#(1,5,6)/(16,5,6)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)#(1,5,1)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)#(1,5)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)#(1,6,1)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)#(1,5)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)#(1,5)
        return s, a, r, d, rtg, timesteps, mask

    return fn

def desc2arrray(train_list):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    #####
    # 初始化存储输出的嵌套列表
    nested_list = []

    # 初始化最大序列长度 n_max
    n_max = 0

    # 遍历每个类别的描述文本
    for class_index, class_descriptions in enumerate(train_list):
        class_outputs = []
        for description in class_descriptions:
            # 使用 Tokenizer 对文本进行编码
            encoded_input = tokenizer(description, return_tensors='pt', padding=True, truncation=True)

            # 将编码后的输入传递给模型
            with torch.no_grad():  # 不计算梯度
                output = model(**encoded_input)

            # 获取最后一层的隐藏状态
            last_hidden_state = output.last_hidden_state.squeeze().detach().numpy()

            # 更新最大序列长度 n_max
            n_max = max(n_max, last_hidden_state.shape[0])

            # 将当前描述的输出添加到类别列表中
            class_outputs.append(last_hidden_state)

        # 将当前类别的输出添加到嵌套列表中
        nested_list.append(class_outputs)

    # 填充嵌套列表以匹配最大序列长度 n_max
    final_nested_list = []
    for class_index, class_outputs in enumerate(nested_list):
        padded_class_outputs = []
        for output in class_outputs:
            # 获取当前输出的序列长度
            n = output.shape[0]
            # 创建一个 n_max * 768 的零矩阵
            padded_output = np.zeros((n_max, 768))
            # 将原始输出填充到零矩阵中
            padded_output[:n, :] = output
            # 将填充后的输出添加到类别列表中
            padded_class_outputs.append(padded_output.tolist())
        # 将填充后的类别列表添加到最终嵌套列表中
        final_nested_list.append(padded_class_outputs)

    # 验证嵌套列表的格式
    print("Nested list shape:", len(final_nested_list), len(final_nested_list[0]), len(final_nested_list[0][0]), len(final_nested_list[0][0][0]))
    return final_nested_list

def desc2arraynew(train_list):
    """
    使用 bert_encoder 对 train_list 中的描述进行编码，并返回嵌套列表形式的结果。

    Args:
        train_list (list): 每个任务的描述列表。
        save_path (str): bert_encoder 模型的保存路径。
        device (str): 设备（默认 'cuda'）。

    Returns:
        list: 嵌套列表，每个任务的描述向量。
    """
    # 加载 bert_encoder 模型
    cur_dir = os.getcwd()
    device='cuda'
    save_path = os.path.join(cur_dir, 'model_saved/')
    model_save_path = os.path.join(save_path, "bert_encoder.pth")
    checkpoint = torch.load(model_save_path)
    bert_encoder = descriptionsEecoder(768, 128).to(device)
    bert_encoder.load_state_dict(checkpoint['bert_encoder_state_dict'])
    bert_encoder.eval()

    # 加载预训练的 BERT 模型
    bert_model_path = "./models/bert-base-uncased"
    bert = BertModel.from_pretrained(bert_model_path).to(device)
    bert.eval()

    # 初始化存储输出的嵌套列表
    nested_list = []

    # 将所有描述文本展开为一个列表
    all_descriptions = [desc for class_descriptions in train_list for desc in class_descriptions]

    # 使用 BERT 对所有描述进行一次性编码
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    with torch.no_grad():
        inputs = tokenizer(all_descriptions, return_tensors="pt", padding=True, truncation=True).to(device)
        bert_outputs = bert(**inputs).last_hidden_state[:, 0, :]  # 提取 [CLS] token 的输出 [num_descriptions, 768]

    # 遍历每个任务的描述文本
    with torch.no_grad():
        start_idx = 0
        for class_descriptions in train_list:
            class_outputs = []
            for _ in class_descriptions:
                # 从 BERT 输出中提取对应的描述向量
                description_vector = bert_outputs[start_idx].unsqueeze(0).to(device)  # [1, 768]
                start_idx += 1

                # 使用 bert_encoder 对描述进行编码
                encoded_output = bert_encoder(description_vector)  # [1, 128]
                class_outputs.append(encoded_output.squeeze(0).cpu().numpy())  # 转换为 numpy 数组

            # 将当前任务的输出添加到嵌套列表中
            nested_list.append(class_outputs)

    return nested_list

    return nested_list

def desc2arrrayMOCO(train_list):
    #tokenizer = BertTokenizer.from_pretrained(model_path)
    #model = BertModel.from_pretrained(model_path)
    #####
    # 初始化存储输出的嵌套列表
    nested_list = []

    # 初始化最大序列长度 n_max
    n_max = 0
    lengths = []  # 用于记录每个 class_descriptions 的长度
    # 遍历每个类别的描述文本
    des_all=[]
    for class_index, class_descriptions in enumerate(train_list):
        # cls_outputs = []
        # for description in class_descriptions:
        #     # 使用 Tokenizer 对文本进行编码
        #     #encoded_input = tokenizer(description, return_tensors='pt', padding=True, truncation=True)

        #     # 将编码后的输入传递给模型
        #     # with torch.no_grad():  # 不计算梯度
        #     #     output = model(**encoded_input)

        #     # 获取最后一层的隐藏状态
        #     #cls_output = output.last_hidden_state[:, 0, :].detach().numpy() # CLS token的输出

        #     # 将当前描述的输出添加到类别列表中
        #     #cls_outputs.append(cls_output)
        #     print('description:',description)
        #     cls_outputs=use_moco(description)
        #     print('cls_outputs:',cls_outputs)
        #     cls_outputs=cls_outputs.tolist()
        # print('class_descriptions',class_descriptions)
        lengths.append(len(class_descriptions))  # 记录当前类别的描述数量
        des_all.extend(class_descriptions)
    cls_outputs=use_moco(des_all)
    start_idx = 0
    nested_list = []
    for length in lengths:
        end_idx = start_idx + length
        nested_list.append(cls_outputs[start_idx:end_idx])  # 分割 cls_outputs
        start_idx = end_idx

    # 返回最终的嵌套列表
    return nested_list

def desc2arrrayCLS(train_list):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    #####
    # 初始化存储输出的嵌套列表
    nested_list = []

    # 初始化最大序列长度 n_max
    n_max = 0

    # 遍历每个类别的描述文本
    des_all=[]
    for class_index, class_descriptions in enumerate(train_list):
        cls_outputs = []
        for description in class_descriptions:
            # 使用 Tokenizer 对文本进行编码
            encoded_input = tokenizer(description, return_tensors='pt', padding=True, truncation=True)

            # 将编码后的输入传递给模型
            with torch.no_grad():  # 不计算梯度
                output = model(**encoded_input)

            # 获取最后一层的隐藏状态
            cls_output = output.last_hidden_state[:, 0, :].detach().numpy() # CLS token的输出

            # 将当前描述的输出添加到类别列表中
            des_all.append(cls_output)
        nested_list.append(des_all)

    
    return nested_list

def get_descriptions(descriptions_list, info, variant):
    print('enter-prompt-utils_get_descriptions')
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_episodes, max_len = variant['prompt_episode'], variant['prompt_length']

    def fn(sample_size=1):
        print('enter-prompt-utils_get_descriptions_fn')
        # random sample prompts with fixed length (prompt-length) in num episodes (prompt-episode)
        #print('sample_size:',sample_size)#第一次是1，后面是16
        batch_inds = np.random.choice(
            np.arange(len(descriptions_list)),#40
            size=int(num_episodes*sample_size),#第一次是1*1，后面是1*16这两个变量是什么东西
            replace=True,
            # p=p_sample,  # reweights so we sample according to timesteps
        )

        descriptions_token = []
        #print('sample_size',sample_size)
        for i in range(int(num_episodes*sample_size)):
            #print('choose:',int(batch_inds[i]))
            descriptions = np.array(descriptions_list[int(batch_inds[i])]) # random select traj
            #descriptions_token.append((descriptions).reshape(1, -1, 768))#这是原来的
            descriptions_token.append((descriptions).reshape(1, -1, 128))
        #print('descriptions_token',descriptions_token)
        # if sample_size!=1:
        #     descriptions_token = torch.tensor(descriptions_token, dtype=torch.float32) 
        #     descriptions_token = torch.stack(descriptions_token, dim=0).to(dtype=torch.float32, device=device)
        descriptions_token = torch.from_numpy(np.concatenate(descriptions_token, axis=0)).to(dtype=torch.float32, device=device)#(1,n,728)/(16,n,728)
        #print('descriptions_token.shape',descriptions_token.shape)
        #input()
        return descriptions_token

    return fn

def get_descriptions_batch(trajectories_list, prompt_trajectories_list, info, variant, train_env_name_list):
    print('enter-prompt-utils_get_descriptions_batch')
    per_env_batch_size = variant['batch_size']#16什么意思？？

    def fn(batch_size=per_env_batch_size):
        print('enter-prompt-utils_get_descriptions_batch_fn')
        descriptions_list = []
        s_list, a_list, r_list, d_list, rtg_list, timesteps_list, mask_list = [], [], [], [], [], [], []
        for env_id, env_name in enumerate(train_env_name_list):#里面就两个环境
            if prompt_trajectories_list is not None and len(prompt_trajectories_list) > 0:
                print(env_id)
                get_descriptions_fn = get_descriptions(prompt_trajectories_list[env_id], info[env_name], variant) #返回一个函数，这个函数针对特定的环境，返回一个prompt，这个prompt中s的shape是(1,5,20)
            else:
                input('no prompt data')
                #get_prompt_fn = get_prompt(trajectories_list[env_id], info[env_name], variant)

            #        return s, a, r, d, rtg, timesteps, mask (1,5,20),(1,5,6)等

            # return fn
            get_batch_fn = get_batch(trajectories_list[env_id], info[env_name], variant) #针对特定的环境，返回16个轨迹的数据
            descriptions= flatten_descriptions(get_descriptions_fn(batch_size), batch_size) #把batch_size赋值给了sample_size，从5个prompt轨迹中随机选择16个轨迹
            descriptions_list.append(descriptions)
            print('1')
            batch = get_batch_fn(batch_size=batch_size)
            s, a, r, d, rtg, timesteps, mask = batch
            if variant['no_r']:
                r = torch.zeros_like(r)
            if variant['no_rtg']:
                rtg = torch.zeros_like(rtg)
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            d_list.append(d)
            rtg_list.append(rtg)
            timesteps_list.append(timesteps)
            mask_list.append(mask)
        descriptions = torch.cat(descriptions_list, dim=0)
        print('3')
        s, a, r, d = torch.cat(s_list, dim=0), torch.cat(a_list, dim=0), torch.cat(r_list, dim=0), torch.cat(d_list, dim=0)
        rtg, timesteps, mask = torch.cat(rtg_list, dim=0), torch.cat(timesteps_list, dim=0), torch.cat(mask_list, dim=0)
        batch = s, a, r, d, rtg, timesteps, mask #(num_env*batch_size, seq_len, dim)
        #print('descriptions.shape',descriptions.shape)
        #print('s.shape',s.shape)
        return descriptions, batch
    return fn

def get_prompt_batch(trajectories_list, prompt_trajectories_list, info, variant, train_env_name_list):
    print('enter-prompt-utils_get_prompt_batch')
    per_env_batch_size = variant['batch_size']#16什么意思？？

    def fn(batch_size=per_env_batch_size):
        print('enter-prompt-utils_get_prompt_batch_fn')
        p_s_list, p_a_list, p_r_list, p_d_list, p_rtg_list, p_timesteps_list, p_mask_list = [], [], [], [], [], [], []
        s_list, a_list, r_list, d_list, rtg_list, timesteps_list, mask_list = [], [], [], [], [], [], []
        for env_id, env_name in enumerate(train_env_name_list):#里面就两个环境
            if prompt_trajectories_list:
                get_prompt_fn = get_prompt(prompt_trajectories_list[env_id], info[env_name], variant) #返回一个函数，这个函数针对特定的环境，返回一个prompt，这个prompt中s的shape是(1,5,20)
            else:
                get_prompt_fn = get_prompt(trajectories_list[env_id], info[env_name], variant)

            #        return s, a, r, d, rtg, timesteps, mask (1,5,20),(1,5,6)等

            # return fn
            get_batch_fn = get_batch(trajectories_list[env_id], info[env_name], variant) #针对特定的环境，返回16个轨迹的数据
            prompt = flatten_prompt(get_prompt_fn(batch_size), batch_size) #把batch_size赋值给了sample_size，从5个prompt轨迹中随机选择16个轨迹
            p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask = prompt
            p_s_list.append(p_s)
            p_a_list.append(p_a)
            p_r_list.append(p_r)
            p_d_list.append(p_d)
            p_rtg_list.append(p_rtg)
            p_timesteps_list.append(p_timesteps)
            p_mask_list.append(p_mask)

            batch = get_batch_fn(batch_size=batch_size)
            s, a, r, d, rtg, timesteps, mask = batch
            if variant['no_r']:
                r = torch.zeros_like(r)
            if variant['no_rtg']:
                rtg = torch.zeros_like(rtg)
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            d_list.append(d)
            rtg_list.append(rtg)
            timesteps_list.append(timesteps)
            mask_list.append(mask)

        p_s, p_a, p_r, p_d = torch.cat(p_s_list, dim=0), torch.cat(p_a_list, dim=0), torch.cat(p_r_list, dim=0), torch.cat(p_d_list, dim=0)
        p_rtg, p_timesteps, p_mask = torch.cat(p_rtg_list, dim=0), torch.cat(p_timesteps_list, dim=0), torch.cat(p_mask_list, dim=0)
        s, a, r, d = torch.cat(s_list, dim=0), torch.cat(a_list, dim=0), torch.cat(r_list, dim=0), torch.cat(d_list, dim=0)
        rtg, timesteps, mask = torch.cat(rtg_list, dim=0), torch.cat(timesteps_list, dim=0), torch.cat(mask_list, dim=0)
        prompt = p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask
        batch = s, a, r, d, rtg, timesteps, mask #(num_env*batch_size, seq_len, dim)
        return prompt, batch
    return fn

""" batches """

def get_batch(trajectories, info, variant):
    print('enter-prompt-utils_get_batch')
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    batch_size, K = variant['batch_size'], variant['K']

    def fn(batch_size=batch_size, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )# 随机选择16个轨迹

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            # 随机选择一个步长作为起点
            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            # 提取max_len长度的数据-20
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            # .shape[1] 返回的是该状态序列的时间步长度（即状态序列的列数），也就是当前轨迹在时间维度上的长度。
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device) # TODO: why mask only has several zeros
        # print('s.shape',s.shape)#(16,20,20)
        # print('a.shape',a.shape)#(16,20,6)
        # print('r.shape',r.shape)#(16,20,1)
        # print('d.shape',d.shape)#(16,20)
        # print('rtg.shape',rtg.shape)#(16,21,1)
        # print('timesteps.shape',timesteps.shape)#(16,20)
        # print('mask.shape',mask.shape)#(16,20)
        return s, a, r, d, rtg, timesteps, mask

    return fn


def get_batch_zsq(trajectories, info, variant):#针对特定环境
    num_traj=999
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    action, state, next_state, reward, d = [], [], [], [], []
    for i in range (num_traj):
        traj = trajectories[i]
        state.append(traj['observations'][:].reshape(1, -1, state_dim)) #(1,200,20)
        action.append(traj['actions'][:].reshape(1, -1, act_dim))
        next_state.append(traj['next_observations'][:].reshape(1, -1, state_dim))
        if 'terminals' in traj:
            d.append(traj['terminals'][:].reshape(1, -1))
        else:
            d.append(traj['dones'][:].reshape(1, -1))

    action = torch.from_numpy(np.concatenate(action, axis=0)).to(dtype=torch.float32, device=device) # (999,200,6)
    state = torch.from_numpy(np.concatenate(state, axis=0)).to(dtype=torch.float32, device=device)
    next_state = torch.from_numpy(np.concatenate(next_state, axis=0)).to(dtype=torch.float32, device=device)
    reward = torch.from_numpy(np.concatenate(reward, axis=0)).to(dtype=torch.float32, device=device)
    d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)

    # action (999,200,dim)
    # reward (999,200,1)
    return state, next_state, action, reward, d



def get_batch_finetune(trajectories, info, variant):
    print('enter-prompt-utils_get_batch_finetune')
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    batch_size, K = variant['batch_size'], variant['prompt_length'] # use the same amount of data for funetuning

    def fn(batch_size=batch_size, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            si = max(0, traj['rewards'].shape[0] - max_len -1) # select the last traj with length max_len

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device) # TODO: why mask only has several zeros

        return s, a, r, d, rtg, timesteps, mask

    return fn

""" data processing """

def process_total_data_mean(trajectories, mode):
    print('enter-prompt-utils_process_total_data_mean')
    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))#储存每个轨迹的长度
        returns.append(path['rewards'].sum())#储存每个轨迹的价值
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    #print('states.shape',states.shape)(799200, 20)
    #print('traj_lens.shape',traj_lens.shape)(3996,)
    #print('returns.shape',returns.shape)(3996,)一共3996个轨迹
    return state_mean, state_std


def process_dataset(trajectories, mode, env_name, dataset, pct_traj):
    print('enter-prompt-utils_process_dataset')
    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    # states_temp=np.array(states)
    # traj_lens_temp=np.array(traj_lens)
    # returns_temp=np.array(returns)
    # print("states:",states_temp.shape)(999,200,20)200个时间步长
    # print("traj_lens:",traj_lens_temp.shape)(999,)
    # print("return:",returns_temp.shape)(999,)
    # input()
    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    # print('p:',pct_traj)1
    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]#同dt，取奖励和最大的几条轨迹

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    reward_info = [np.mean(returns), np.std(returns), np.max(returns), np.min(returns)]
    #print('num_traj',num_trajectories)998???
    return trajectories, num_trajectories, sorted_inds, p_sample, state_mean, state_std, reward_info


def load_descriptions(env_name_list, data_save_path,desc_mode,args):
    descriptions_list = []
    for env_name in env_name_list:
        desc_path = data_save_path+f'/{args.env}/{env_name}-desc.py'
        with open(desc_path, 'r', encoding='utf-8') as f:
            desc = f.read()
        
        # 假设文件内容是用换行符分隔的字符串列表
        # 使用 splitlines() 将其转换为列表
        desc_lines = desc.splitlines()
        
        # 取前40个句子
        if desc_mode == 'train':
            descriptions_list.append(desc_lines[:40])
        elif desc_mode == 'test':
            descriptions_list.append(desc_lines[40:])

    #print('descriptions_list:',descriptions_list)
    return descriptions_list


def load_descriptions_new(env_name_list, data_save_path, desc_mode, args):
    descriptions_list = []
    for env_name in env_name_list:
        # 获取文件对应的数字
        task_id = int(env_name.split('-')[-1])  # 从文件名中提取任务 ID
        task_key = id2key[task_id]  # 从 key.py 中获取对应的数字

        # 根据任务编号生成文字描述
        description = f"Please run at the target velocity of {task_key}"

        # 根据模式生成描述
        if desc_mode == 'train':
            descriptions_list.append([description] * 40)  # 添加40句相同的文字描述
        elif desc_mode == 'test':
            descriptions_list.append([description] * 10)  # 测试模式同理

    return descriptions_list

    return descriptions_list
def load_data_prompt(env_name_list, data_save_path, dataset, prompt_mode, args):
    trajectories_list = []
    prompt_trajectories_list = []
    for env_name in env_name_list:#一共2个环境,前进后退
        dataset_path = data_save_path+f'/{args.env}/{env_name}-{dataset}.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        prompt_dataset_path = data_save_path+f'/{args.env}/{env_name}-prompt-{prompt_mode}.pkl'
        with open(prompt_dataset_path, 'rb') as f:
            prompt_trajectories = pickle.load(f)
            # for i, trajectory in enumerate(prompt_trajectories):
            #     print(f"Trajectory {i}:")
            #     for key, value in trajectory.items():
            #         print(f"  Key: {key}, List size: {len(value)}")
            #     print("-" * 50)

            # input()每个key都有200个值
        trajectories_list.append(trajectories)
        prompt_trajectories_list.append(prompt_trajectories)

        
    # print('traj path:')
    # print(dataset_path)
    # print('prompt traj path')
    # print(prompt_dataset_path)
    # input()
    return trajectories_list, prompt_trajectories_list


def process_info(env_name_list, trajectories_list, info, mode, dataset, pct_traj, variant):
    print('enter-prompt-utils_process_info')
    for i, env_name in enumerate(env_name_list):#分别读取了两个环境的训练、测试数据
        trajectories, num_trajectories, sorted_inds, p_sample, state_mean, state_std, reward_info = process_dataset(
            trajectories=trajectories_list[i], mode=mode, env_name=env_name_list[i], dataset=dataset, pct_traj=pct_traj)
        info[env_name]['num_trajectories'] = num_trajectories
        info[env_name]['sorted_inds'] = sorted_inds
        info[env_name]['p_sample'] = p_sample
        info[env_name]['state_mean'] = state_mean
        info[env_name]['state_std'] = state_std
        if variant['average_state_mean']:
            info[env_name]['state_mean'] = variant['total_state_mean']
            info[env_name]['state_std'] = variant['total_state_std']
    return info


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

""" evaluation """

def eval_episodes(target_rew, info, variant, env, env_name):
    print('enter-prompt-utils_eval_epi')
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_eval_episodes = variant['num_eval_episodes']
    mode = variant.get('mode', 'normal')

    def fn(model, descriptions=None):
        returns = []
        for _ in range(num_eval_episodes):
            print('enter-prompt-utils_eval_epi_:',_)
            with torch.no_grad():
                ret, infos = prompt_evaluate_episode_rtg(
                    env,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_rew / scale,
                    mode=mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    descriptions=descriptions,
                    no_r=variant['no_r'],
                    no_rtg=variant['no_rtg'],
                    no_state_normalize=variant['no_state_normalize']                
                    )
            returns.append(ret)
            #input('eval_episode')
        return {
            f'{env_name}_target_{target_rew}_return_mean': np.mean(returns),
            f'{env_name}_target_{target_rew}_return_std': np.std(returns),
            }
    return fn

