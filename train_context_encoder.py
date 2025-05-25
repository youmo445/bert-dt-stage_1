from ast import parse
import sys
sys.path.append('envs/mujoco-control-envs/mujoco_control_envs/tp_envs/rand_param_envs/rand_param_envs/gym')
import gym
import numpy as np
import torch
import wandb
# 指定本地模型和分词器的路径
model_path = './models/bert-base-uncased'
import argparse
import pickle
import random
import sys
import time
import itertools
from sklearn.manifold import TSNE
from prompt_dt.prompt_decision_transformer import PromptDecisionTransformer
#from prompt_dt.prompt_seq_trainer import PromptSequenceTrainer
from prompt_dt.prompt_utils import get_env_list, preprocess_descriptions
#from prompt_dt.prompt_utils import get_prompt_batch, get_prompt, get_batch, get_batch_finetune,get_descriptions, get_descriptions_batch
#from prompt_dt.prompt_utils import process_total_data_mean, load_data_prompt, process_info, load_descriptions, desc2arrray, desc2arrrayCLS, load_descriptions_new, desc2arrrayMOCO
#from prompt_dt.prompt_utils import eval_episodes
#from context.model import RNNContextEncoder, RewardDecoder, StateDecoder, descriptionsEecoder
from collections import namedtuple
import json, pickle, os
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
def experiment_mix_env(
        exp_prefix,
        variant,
):
    TRAIN_AND_SAVE = False
    LOAD = True
    EVAL_WORLD_MODEL = True
    LORA_AND_SAVE_1 = False
    LORA_AND_SAVE_2 = False
    SAVE_BERT_PKL = False
    EVAL_BERT_MODEL_1 = False
    EVAL_BERT_MODEL_2 = False
    device = variant['device']
    log_to_wandb = variant['log_to_wandb']

    ######
    # construct train and test environments
    ######
    
    cur_dir = os.getcwd()
    config_save_path = os.path.join(cur_dir, 'config')
    data_save_path = os.path.join(cur_dir, 'data')
    save_path = os.path.join(cur_dir, 'model_saved/')
    if not os.path.exists(save_path): os.mkdir(save_path)

    config_path_dict = {
        'cheetah_vel': "cheetah_vel/cheetah_vel_40.json",
        'cheetah_dir': "cheetah_dir/cheetah_dir_2.json",
        'ant_dir': "ant_dir/ant_dir_50.json",
        'ML1-pick-place-v2': "ML1-pick-place-v2/ML1_pick_place.json",
    }
    
    task_config = os.path.join(config_save_path, config_path_dict[args.env])
    with open(task_config, 'r') as f:
        task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    # X(env='cheetah_vel', total_tasks=40, train_tasks=[0, 1, 3], test_tasks=[2, 7])
    train_env_name_list, test_env_name_list = [], []
    for task_ind in task_config.train_tasks:
        train_env_name_list.append(args.env +'-'+ str(task_ind))
    for task_ind in task_config.test_tasks:
        test_env_name_list.append(args.env +'-'+ str(task_ind))
    # training envs
    info, env_list = get_env_list(train_env_name_list, config_save_path, device)
    # testing envs
    test_info, test_env_list = get_env_list(test_env_name_list, config_save_path, device)

    print(f'Env Info: {info} \n\n Test Env Info: {test_info}\n\n\n')
    print(f'Env List: {env_list} \n\n Test Env List: {test_env_list}')
    #input('Press Enter to continue...')  # 等待用户按下回车键
    ######
    # process train and test datasets
    ######

    K = variant['K']#从varient字典中取出键为k的值赋给K
    batch_size = variant['batch_size']# 16
    pct_traj = variant.get('pct_traj', 1.)#从varient字典中取出键为pct_traj的值赋值，不存在则赋1
    mode = variant.get('mode', 'normal')
    dataset_mode = variant['dataset_mode']
    test_dataset_mode = variant['test_dataset_mode']
    train_prompt_mode = variant['train_prompt_mode']
    test_prompt_mode = variant['test_prompt_mode']

    # load training dataset
    trajectories_list, prompt_trajectories_list = load_data_prompt(train_env_name_list, data_save_path, dataset_mode, train_prompt_mode, args)
    # load testing dataset
    test_trajectories_list, test_prompt_trajectories_list = load_data_prompt(test_env_name_list, data_save_path, test_dataset_mode, test_prompt_mode, args)
    print('traj_list',np.array(trajectories_list).shape)#(2,999)#两个环境，每个环境999条轨迹
    print('prompt_traj_list',np.array(prompt_trajectories_list).shape)#(2,5)
    print('test_traj_list',np.array(test_trajectories_list).shape)#(2,999)


    if variant['average_state_mean']:
        train_total = list(itertools.chain.from_iterable(trajectories_list))
        test_total = list(itertools.chain.from_iterable(test_trajectories_list))
        total_traj_list = train_total + test_total
        #print('train_total',len(train_total))#1998，训练的两个环境，每个环境999条轨迹
        #print('test_total',len(test_total))#1998
        #print('total_traj_list',len(total_traj_list))#训练轨迹和测试轨迹的总条数3996，即两个环境共3996条，每条轨迹20个时间步长
        total_state_mean, total_state_std= process_total_data_mean(total_traj_list, mode)
        variant['total_state_mean'] = total_state_mean
        variant['total_state_std'] = total_state_std

    # process train info
    info = process_info(train_env_name_list, trajectories_list, info, mode, dataset_mode, pct_traj, variant)
    #print('info',info)
    #input('Press Enter to continue...')  # 等待用户按下回车键
    # process test info
    test_info = process_info(test_env_name_list, test_trajectories_list, test_info, mode, test_dataset_mode, pct_traj, variant)
    
    print('ending process train and test datasets-----------------')
    

    
    # 假设 contextEncoder、stateDecoder 和 rewardDecoder 是模型实例
    state_dim = test_env_list[0].observation_space.shape[0]
    action_dim = test_env_list[0].action_space.shape[0]
    context_dim=128
    context_hidden_dim=256
    iteration=1000
    bert_dim = 768
    m=100
    def choosetask(num):
        return random.randint(0, num-1)
    def choosedata(M):
        return trajectories_list[M]
    def choose2trajectories(D):
        indices = random.sample(range(len(D)), 2)
        trajectory1, trajectory2 = D[indices[0]], D[indices[1]]
        return trajectory1, trajectory2
    context_encoder =RNNContextEncoder(state_dim, action_dim, context_dim, context_hidden_dim).to(device)
    state_decoder = StateDecoder(state_dim, action_dim, context_dim, context_hidden_dim).to(device)
    reward_decoder = RewardDecoder(state_dim, action_dim, context_dim, context_hidden_dim).to(device)
    bert_encoder = descriptionsEecoder(bert_dim,context_dim).to(device)
    
    if  TRAIN_AND_SAVE:
        # 初始化用于记录损失的列表
        optimizer = torch.optim.Adam([*context_encoder.parameters(), *reward_decoder.parameters(),*state_decoder.parameters()], lr=0.0003)
        losses = []
        context_encoder.train()  # 设置模型为训练模式
        state_decoder.train()
        reward_decoder.train()
        for i in range(iteration):
            batch_loss = 0  # 初始化每个 epoch 的总损失
            optimizer.zero_grad()  # 在每个 iteration 开始时清零梯度
            T_batch = []
            T_star_batch = []

            for j in range(m):  # 遍历 batch_size=m 的所有批次
                # 随机选择任务
                M = choosetask(35)
                # 获取任务对应的轨迹数据
                D = choosedata(M)
                # 随机选择两条轨迹
                T, T_star = choose2trajectories(D)
                T_batch.append(T)
                T_star_batch.append(T_star)
                # 从轨迹 T 中提取状态、动作和奖励
            # 将批次中的状态、动作和奖励组织成张量
            states = torch.tensor([T['observations'] for T in T_batch], dtype=torch.float32).to(device)  # [m, seq_len, state_dim]
            actions = torch.tensor([T['actions'] for T in T_batch], dtype=torch.float32).to(device)  # [m, seq_len, action_dim]
            rewards = torch.tensor([T['rewards'] for T in T_batch], dtype=torch.float32).to(device)  # [m, seq_len, 1]

            # 转换为 [seq_len, m, dim] 以适配 RNNContextEncoder 的输入
            states = states.permute(1, 0, 2)  # [seq_len, m, state_dim]
            actions = actions.permute(1, 0, 2)  # [seq_len, m, action_dim]
            rewards = rewards.permute(1, 0, 2)  # [seq_len, m, 1]

            # 输入状态、动作和奖励到 contextEncoder，生成上下文向量 Z
            Z = context_encoder(states, actions, rewards)  # [m, context_dim]
            # 将轨迹 T_star 中的所有时间步组织成批次
            obs_batch = torch.tensor([T_star['observations'] for T_star in T_star_batch], dtype=torch.float32).reshape(-1, states.shape[-1]).to(device)  # [m*seq_len, state_dim]
            action_batch = torch.tensor([T_star['actions'] for T_star in T_star_batch], dtype=torch.float32).reshape(-1, actions.shape[-1]).to(device)  # [m*seq_len, action_dim]
            next_obs_batch = torch.tensor([T_star['next_observations'] for T_star in T_star_batch], dtype=torch.float32).reshape(-1, states.shape[-1]).to(device)  # [m*seq_len, state_dim]
            reward_batch = torch.tensor([T_star['rewards'] for T_star in T_star_batch], dtype=torch.float32).reshape(-1, 1).to(device)  # [m*seq_len, 1]
            done_batch = torch.tensor([T_star['terminals'] for T_star in T_star_batch], dtype=torch.float32).reshape(-1).to(device)  # [m*seq_len]


            # 复制 Z 以匹配批次大小
            Z_batch = Z.repeat_interleave(200, dim=0)  # [m*seq_len, context_dim]

            # 输入批次数据到 stateDecoder 和 rewardDecoder
            predicted_next_obs_batch = state_decoder(obs_batch, action_batch, Z_batch)  # [m*seq_len, state_dim]
            predicted_reward_batch = reward_decoder(obs_batch, action_batch, Z_batch)  # [m*seq_len, 1]

            # 计算批次的状态估计损失和奖励估计损失
            state_loss = torch.nn.functional.mse_loss(predicted_next_obs_batch, next_obs_batch)
            reward_loss = torch.nn.functional.mse_loss(predicted_reward_batch, reward_batch)

            # 累加批次损失
            batch_loss = state_loss + reward_loss
            batch_loss = batch_loss / m  # 平均损失
            batch_loss.backward()  # 累积梯度
            optimizer.step()
            # 记录损失
            avg_loss = batch_loss.item()  # 将 Tensor 转换为数值
            losses.append(avg_loss)  # 添加到损失列表

            # 打印当前 epoch 的损失
            print(f"Epoch {i + 1}/{iteration}, Loss: {avg_loss}")


        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, iteration + 1), losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid()
        plt.show()

        # 保存模型
        model_save_path = os.path.join(save_path, "context_encoder.pth")
        torch.save({
            'context_encoder_state_dict': context_encoder.state_dict(),
            'state_decoder_state_dict': state_decoder.state_dict(),
            'reward_decoder_state_dict': reward_decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses
        }, model_save_path)
        print(f"Model saved to {model_save_path}")
        # 评估部分
    if LOAD:
        # 加载模型
        optimizer = torch.optim.Adam([*context_encoder.parameters(), *reward_decoder.parameters(),*state_decoder.parameters()], lr=0.0003)
        #optimizer = torch.optim.Adam([*context_encoder.parameters(), *reward_decoder.parameters()], lr=0.0003)
        model_save_path = os.path.join(save_path, "context_encoder.pth")
        checkpoint = torch.load(model_save_path)
        context_encoder.load_state_dict(checkpoint['context_encoder_state_dict'])
        state_decoder.load_state_dict(checkpoint['state_decoder_state_dict'])
        reward_decoder.load_state_dict(checkpoint['reward_decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        losses = checkpoint['losses']
        print(f"Model loaded from {model_save_path}")
    if EVAL_WORLD_MODEL:
        with torch.no_grad():
            context_encoder.eval()
            state_decoder.eval()
            reward_decoder.eval()

            # 对原先的 35 个任务进行降维
            all_train_Z = []
            train_labels = []

            for train_env_idx, train_trajectories in enumerate(trajectories_list):
                states = torch.tensor([traj['observations'] for traj in train_trajectories], dtype=torch.float32).to(device)  # [num_traj, seq_len, state_dim]
                actions = torch.tensor([traj['actions'] for traj in train_trajectories], dtype=torch.float32).to(device)  # [num_traj, seq_len, action_dim]
                rewards = torch.tensor([traj['rewards'] for traj in train_trajectories], dtype=torch.float32).to(device)  # [num_traj, seq_len, 1]

                # 转换为 [seq_len, num_traj, dim]
                states = states.permute(1, 0, 2)
                actions = actions.permute(1, 0, 2)
                rewards = rewards.permute(1, 0, 2)

                # 生成上下文向量 Z
                Z = context_encoder(states, actions, rewards)  # [num_traj, context_dim]
                all_train_Z.append(Z.cpu().numpy())
                train_labels.extend([train_env_idx] * Z.shape[0])

            # 将所有训练任务的 Z 转换为 numpy 数组
            all_train_Z = np.concatenate(all_train_Z, axis=0)  # [num_train_traj, context_dim]

            # 使用 T-SNE 对训练任务的 Z 进行降维
            tsne_train = TSNE(n_components=2, random_state=42)
            train_Z_2d = tsne_train.fit_transform(all_train_Z)  # [num_train_Z, 2]

            from matplotlib import cm
            
            # 生成 35 种离散颜色
            colors = cm.get_cmap('tab20', 35)
            
            # 绘制训练任务的 T-SNE 降维结果
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(train_Z_2d[:, 0], train_Z_2d[:, 1], c=train_labels, cmap=colors, alpha=0.7)
            plt.colorbar(scatter, label="Task Index")
            plt.title("T-SNE Visualization of Train Context Vectors")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.grid()
            plt.show()
            all_Z = []
            labels = []
            eval_losses = []

            for test_env_idx, test_trajectories in enumerate(test_trajectories_list):
                states = torch.tensor([traj['observations'] for traj in test_trajectories], dtype=torch.float32).to(device)  # [num_traj, seq_len, state_dim]
                actions = torch.tensor([traj['actions'] for traj in test_trajectories], dtype=torch.float32).to(device)  # [num_traj, seq_len, action_dim]
                rewards = torch.tensor([traj['rewards'] for traj in test_trajectories], dtype=torch.float32).to(device)  # [num_traj, seq_len, 1]

                # 转换为 [seq_len, num_traj, dim]
                states = states.permute(1, 0, 2)
                actions = actions.permute(1, 0, 2)
                rewards = rewards.permute(1, 0, 2)

                # 生成上下文向量 Z
                Z = context_encoder(states, actions, rewards)  # [999, 128]
                all_Z.append(Z.cpu().numpy())
                labels.extend([test_env_idx] * Z.shape[0])

                # 计算评估损失
                obs_batch = torch.tensor([traj['observations'] for traj in test_trajectories], dtype=torch.float32).reshape(-1, states.shape[-1]).to(device)
                action_batch = torch.tensor([traj['actions'] for traj in test_trajectories], dtype=torch.float32).reshape(-1, actions.shape[-1]).to(device)
                next_obs_batch = torch.tensor([traj['next_observations'] for traj in test_trajectories], dtype=torch.float32).reshape(-1, states.shape[-1]).to(device)
                reward_batch = torch.tensor([traj['rewards'] for traj in test_trajectories], dtype=torch.float32).reshape(-1, 1).to(device)

                Z_batch = Z.repeat_interleave(200, dim=0)  # [num_traj*seq_len, context_dim]

                predicted_next_obs_batch = state_decoder(obs_batch, action_batch, Z_batch)
                predicted_reward_batch = reward_decoder(obs_batch, action_batch, Z_batch)

                state_loss = torch.nn.functional.mse_loss(predicted_next_obs_batch, next_obs_batch)
                reward_loss = torch.nn.functional.mse_loss(predicted_reward_batch, reward_batch)
                eval_loss = state_loss.item() + reward_loss.item()

                eval_losses.append(eval_loss / len(test_trajectories))

            # 将所有 Z 转换为 numpy 数组
            all_Z = np.concatenate(all_Z, axis=0)  # [999*5, context_dim]

            # 使用 T-SNE 对 Z 进行降维
            tsne = TSNE(n_components=2, random_state=42)
            Z_2d = tsne.fit_transform(all_Z)  # [num_Z, 2]

            # 绘制 T-SNE 降维结果
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)

            # 添加颜色条
            plt.colorbar(scatter, label="Environment Index")

            # 添加自定义图例
            unique_labels = np.unique(labels)  # 获取唯一的环境索引
            legend_labels = [f"Env {int(label)}" for label in unique_labels]  # 创建图例标签
            for i, label in enumerate(unique_labels):
                plt.scatter([], [], color=plt.cm.tab10(i / len(unique_labels)), label=legend_labels[i])  # 空点用于图例

            plt.legend(title="Environments", loc="best")  # 添加图例

            # 设置标题和坐标轴标签
            plt.title("T-SNE Visualization of Context Vectors")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.grid(True)
            plt.show()
    if SAVE_BERT_PKL:
        desc_dir = "C:/Users/16340/Desktop/DT/bert-dt-stage_1/data/cheetah_vel"
        output_path = "C:/Users/16340/Desktop/DT/bert-dt-stage_1/preprocessed_vectors.pkl"
        bert_model_path = "./models/bert-base-uncased"
        preprocess_descriptions(desc_dir, output_path, bert_model_path, device='cuda')
    if LORA_AND_SAVE_1:
        desc_dir = "C:/Users/16340/Desktop/DT/bert-dt-stage_1/data/cheetah_vel"
        output_path = "C:/Users/16340/Desktop/DT/bert-dt-stage_1/preprocessed_vectors.pkl"
        bert_model_path = "./models/bert-base-uncased"
        # preprocess_descriptions(desc_dir, output_path, bert_model_path, device='cuda')
        # input('Press Enter to continue...')  # 等待用户按下回车键
        with open("C:/Users/16340/Desktop/DT/bert-dt-stage_1/preprocessed_vectors.pkl", 'rb') as f:
            preprocessed_vectors = pickle.load(f)
        optimizer = torch.optim.Adam([*bert_encoder.parameters()], lr=0.0003)
        def sampleN(N):
            indices = random.sample(train_env_name_list, N)  # 随机选择N个不同环境的索引
            T_batch = []
            l_batch = []
            for task_name in indices:
                #print(f"Task Name: {task_name}")
                task_idx = train_env_name_list.index(task_name)  # 根据任务名找到其在 train_env_name_list 中的索引
                # 获取对应任务的轨迹
                #print(f"Task Index: {task_idx}")
                env_trajectories = trajectories_list[task_idx]
                trajectory = random.choice(env_trajectories)  # 从该任务中随机选择一条轨迹
                T_batch.append(trajectory)
                # 从预处理向量中加载对应描述的向量
                desc_file = f"{task_name}-desc.py"
                vectors = preprocessed_vectors[desc_file]
                #print(f"Vectors.shape: {len(vectors)}")
                chosen_vector = random.choice(vectors)  # 随机选择一个描述的向量
                l_batch.append(torch.tensor(chosen_vector, dtype=torch.float32).to(device))
            return T_batch, torch.stack(l_batch)

        iteration = 10000
        N= 35
        losses = []
        for i in range(iteration):
            optimizer.zero_grad()
            T_batch, l_batch = sampleN(N)
            # 将 T_batch 中的所有轨迹的 observations 合并为一个张量
            states = torch.tensor(
                [traj['observations'] for traj in T_batch], dtype=torch.float32
            ).permute(1, 0, 2).to(device)  # [seq_len, N, state_dim]
            
            actions = torch.tensor(
                [traj['actions'] for traj in T_batch], dtype=torch.float32
            ).permute(1, 0, 2).to(device)  # [seq_len, N, action_dim]
            
            rewards = torch.tensor(
                [traj['rewards'] for traj in T_batch], dtype=torch.float32
            ).permute(1, 0, 2).to(device)  # [seq_len, N, 1]
            
            # 使用 context_encoder 生成上下文向量
            with torch.no_grad():
                context_encoder.eval()
                T_batch = context_encoder(states, actions, rewards)  # [N, context_dim]
            
            bert_encoder.train()
            l_batch = bert_encoder(l_batch)  # [N, context_dim]
            print(f"l_batch.shape: {l_batch.shape}")
            # 假设 T_batch 和 l_batch 的形状分别为 [N, context_dim] 和 [N, 128]
            # 计算余弦相似性矩阵
            similarity_matrix = F.cosine_similarity(T_batch.unsqueeze(1), l_batch.unsqueeze(0), dim=-1)  # [N, N]
            
            # 构造对比学习的目标
            N = T_batch.shape[0]
            labels = torch.arange(N).to(device)  # 正样本的索引 [0, 1, 2, ..., N-1]
            
            # 计算 InfoNCE 损失
            temperature = 0.1  # 温度参数
            similarity_matrix /= temperature  # 缩放相似性矩阵
            contrastive_loss = F.cross_entropy(similarity_matrix, labels)
            losses.append(contrastive_loss.item())  # 记录损失
            # 打印损失
            print(f"Contrastive Loss: {i,contrastive_loss.item()}")
            
            # 反向传播并优化
            contrastive_loss.backward()
            optimizer.step()
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, iteration + 1), losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid()
        plt.show()
        model_save_path = os.path.join(save_path, "bert_encoder.pth")
        torch.save({
            'bert_encoder_state_dict': bert_encoder.state_dict()
        }, model_save_path)
        print(f"Model saved to {model_save_path}")
    if LORA_AND_SAVE_2:
        bert_model_path = "./models/bert-base-uncased"
        optimizer = torch.optim.Adam([*bert_encoder.parameters()], lr=0.0003)
    
        # 从 cheetah_vel_descriptions.py 加载任务编号和速度的对应关系
        from data.cheetah_vel_descriptions import id2key
    
        # 加载预训练的 BERT 模型
        bert = BertModel.from_pretrained(bert_model_path).to(device)
        bert.eval()  # 设置 BERT 为评估模式
    
        # 预处理所有任务的描述语句并存储为 BERT 向量
        preprocessed_descriptions = {}
        tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        with torch.no_grad():
            descriptions = []
            task_names = []
            for task_name in train_env_name_list:
                task_id = int(task_name.split('-')[-1])  # 假设任务名格式为 "cheetah_vel-X"
                velocity = id2key[task_id]  # 从 id2key 获取对应的速度
                description = f"Please run at the target velocity of {velocity}"
                descriptions.append(description)
                task_names.append(task_name)
    
            # 一次性对所有描述语句进行编码
            inputs = tokenizer(descriptions, return_tensors="pt", padding=True, truncation=True).to(device)
            bert_outputs = bert(**inputs).last_hidden_state[:, 0, :]  # 提取 [CLS] token 的输出
    
            # 存储每个任务的描述向量
            for i, task_name in enumerate(task_names):
                preprocessed_descriptions[task_name] = bert_outputs[i].squeeze(0)  # 存储为 [768]
    
        def sampleN(N):
            indices = random.sample(train_env_name_list, N)  # 随机选择 N 个不同环境的索引
            T_batch = []
            l_batch = []
            for task_name in indices:
                task_idx = train_env_name_list.index(task_name)  # 根据任务名找到其在 train_env_name_list 中的索引
                env_trajectories = trajectories_list[task_idx]
                trajectory = random.choice(env_trajectories)  # 从该任务中随机选择一条轨迹
                T_batch.append(trajectory)
    
                # 从预处理好的描述向量中加载对应的向量
                description_vector = preprocessed_descriptions[task_name]
                l_batch.append(description_vector)
    
            return T_batch, torch.stack(l_batch)
    
        iteration = 2000
        N = 35
        losses = []
        for i in range(iteration):
            optimizer.zero_grad()
            T_batch, l_batch = sampleN(N)
    
            # 将 T_batch 中的所有轨迹的 observations 合并为一个张量
            states = torch.tensor(
                [traj['observations'] for traj in T_batch], dtype=torch.float32
            ).permute(1, 0, 2).to(device)  # [seq_len, N, state_dim]
    
            actions = torch.tensor(
                [traj['actions'] for traj in T_batch], dtype=torch.float32
            ).permute(1, 0, 2).to(device)  # [seq_len, N, action_dim]
    
            rewards = torch.tensor(
                [traj['rewards'] for traj in T_batch], dtype=torch.float32
            ).permute(1, 0, 2).to(device)  # [seq_len, N, 1]
    
            # 使用 context_encoder 生成上下文向量
            with torch.no_grad():
                context_encoder.eval()
                T_batch = context_encoder(states, actions, rewards)  # [N, context_dim]
    
            # 使用 bert_encoder 将 BERT 输出的向量转换为目标上下文向量
            bert_encoder.train()
            l_batch = bert_encoder(l_batch.to(device))  # [N, context_dim]
    
            # 计算余弦相似性矩阵
            similarity_matrix = F.cosine_similarity(T_batch.unsqueeze(1), l_batch.unsqueeze(0), dim=-1)  # [N, N]
    
            # 构造对比学习的目标
            labels = torch.arange(N).to(device)  # 正样本的索引 [0, 1, 2, ..., N-1]
    
            # 计算 InfoNCE 损失
            temperature = 0.1  # 温度参数
            similarity_matrix /= temperature  # 缩放相似性矩阵
            contrastive_loss = F.cross_entropy(similarity_matrix, labels)
            losses.append(contrastive_loss.item())  # 记录损失
    
            # 打印损失
            print(f"Contrastive Loss: {i}, {contrastive_loss.item()}")
    
            # 反向传播并优化
            contrastive_loss.backward()
            optimizer.step()
    
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, iteration + 1), losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid()
        plt.show()
    
        # 保存模型
        model_save_path = os.path.join(save_path, "bert_encoder.pth")
        torch.save({
            'bert_encoder_state_dict': bert_encoder.state_dict()
        }, model_save_path)
        print(f"Model saved to {model_save_path}")
    if EVAL_BERT_MODEL_1:
        model_save_path = os.path.join(save_path, "bert_encoder.pth")
        checkpoint = torch.load(model_save_path)
        bert_encoder.load_state_dict(checkpoint['bert_encoder_state_dict'])
        print(f"bert_encoder model loaded from {model_save_path}")
    
        # 加载预处理好的向量
        with open("C:/Users/16340/Desktop/DT/bert-dt-stage_1/preprocessed_vectors.pkl", 'rb') as f:
            preprocessed_vectors = pickle.load(f)
    
        with torch.no_grad():
            bert_encoder.eval()
    
            # 处理训练任务的描述
            train_vectors = []
            for task_name in train_env_name_list:
                desc_file = f"{task_name}-desc.py"
                vectors = preprocessed_vectors[desc_file]
                train_vectors.extend(vectors[-10:])  # 取最后 10 个描述的向量
            train_vectors = torch.tensor(train_vectors, dtype=torch.float32).to(device)
    
            # 处理测试任务的描述
            test_vectors = []
            for task_name in test_env_name_list:
                desc_file = f"{task_name}-desc.py"
                vectors = preprocessed_vectors[desc_file]
                test_vectors.extend(vectors[:50])  # 取前 50 个描述的向量
            test_vectors = torch.tensor(test_vectors, dtype=torch.float32).to(device)
    
            # 打印向量形状
            print(f"train_vectors shape: {train_vectors.shape}")  # [350, context_dim]
            print(f"test_vectors shape: {test_vectors.shape}")  # [250, context_dim]
    
            # 使用 T-SNE 对训练和测试向量降维
            tsne = TSNE(n_components=2, random_state=42)
            train_vectors_2d = tsne.fit_transform(train_vectors.cpu().numpy())
            test_vectors_2d = tsne.fit_transform(test_vectors.cpu().numpy())
    
            # 绘制训练任务的 T-SNE 降维结果
            num_tasks = 35  # 总任务数
            descriptions_per_task = len(train_vectors) // num_tasks  # 每个任务的描述数量
            train_labels = [i // descriptions_per_task for i in range(len(train_vectors))]
    
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(train_vectors_2d[:, 0], train_vectors_2d[:, 1], c=train_labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label="Task Index")
            plt.title("T-SNE Visualization of Train Descriptions")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.grid()
            plt.show()
    
            # 绘制测试任务的 T-SNE 降维结果
            test_labels = [i // 50 for i in range(len(test_vectors))]  # 假设每个任务有 50 个描述
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(test_vectors_2d[:, 0], test_vectors_2d[:, 1], c=test_labels, cmap='tab10', alpha=0.7)
            plt.colorbar(scatter, label="Task Index")
            plt.title("T-SNE Visualization of Test Descriptions")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.grid()
            plt.show()
    if EVAL_BERT_MODEL_2:
        model_save_path = os.path.join(save_path, "bert_encoder.pth")
        checkpoint = torch.load(model_save_path)
        bert_encoder.load_state_dict(checkpoint['bert_encoder_state_dict'])
        print(f"bert_encoder model loaded from {model_save_path}")
    
        # 从 cheetah_vel_descriptions.py 加载任务编号和速度的对应关系
        from data.cheetah_vel_descriptions import id2key
    
        # 加载预训练的 BERT 模型
        bert_model_path = "./models/bert-base-uncased"
        bert = BertModel.from_pretrained(bert_model_path).to(device)
        bert.eval()  # 设置 BERT 为评估模式
    
        # 预处理所有任务的描述语句并存储为 BERT 向量
        preprocessed_descriptions = {}
        tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        with torch.no_grad():
            descriptions = []
            task_names = []
            for task_name in train_env_name_list + test_env_name_list:
                task_id = int(task_name.split('-')[-1])  # 假设任务名格式为 "cheetah_vel-X"
                velocity = id2key[task_id]  # 从 id2key 获取对应的速度
                description = f"Please run at the target velocity of {velocity}"
                descriptions.append(description)
                task_names.append(task_name)
    
            # 一次性对所有描述语句进行编码
            inputs = tokenizer(descriptions, return_tensors="pt", padding=True, truncation=True).to(device)
            bert_outputs = bert(**inputs).last_hidden_state[:, 0, :]  # 提取 [CLS] token 的输出
    
            # 存储每个任务的描述向量
            for i, task_name in enumerate(task_names):
                preprocessed_descriptions[task_name] = bert_outputs[i].squeeze(0)  # 存储为 [768]
    
        # 处理训练任务的描述
        train_vectors = []
        train_labels = []
        for task_name in train_env_name_list:
            description_vector = preprocessed_descriptions[task_name]
            train_vectors.append(description_vector)
            train_labels.append(train_env_name_list.index(task_name))  # 使用任务索引作为标签
        train_vectors = torch.stack(train_vectors).to(device)  # [num_train_tasks, 768]
    
        # 处理测试任务的描述
        test_vectors = []
        test_labels = []
        for task_name in test_env_name_list:
            description_vector = preprocessed_descriptions[task_name]
            test_vectors.append(description_vector)
            test_labels.append(test_env_name_list.index(task_name))  # 使用任务索引作为标签
        test_vectors = torch.stack(test_vectors).to(device)  # [num_test_tasks, 768]
    
        # 使用 bert_encoder 对描述向量进行降维
        with torch.no_grad():
            bert_encoder.eval()
            train_vectors_encoded = bert_encoder(train_vectors)  # [num_train_tasks, context_dim]
            test_vectors_encoded = bert_encoder(test_vectors)  # [num_test_tasks, context_dim]
    
        # 动态调整 perplexity 参数
        train_perplexity = min(30, len(train_vectors_encoded) - 1)
        test_perplexity = min(30, len(test_vectors_encoded) - 1)

        # 使用 T-SNE 对训练和测试向量降维
        tsne = TSNE(n_components=2, perplexity=train_perplexity, random_state=42)
        train_vectors_2d = tsne.fit_transform(train_vectors_encoded.cpu().numpy())

        tsne = TSNE(n_components=2, perplexity=test_perplexity, random_state=42)
        test_vectors_2d = tsne.fit_transform(test_vectors_encoded.cpu().numpy())
    
        # 绘制训练任务的 T-SNE 降维结果
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(train_vectors_2d[:, 0], train_vectors_2d[:, 1], c=train_labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label="Task Index")
        plt.title("T-SNE Visualization of Train Descriptions")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid()
        plt.show()
    
        # 绘制测试任务的 T-SNE 降维结果
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(test_vectors_2d[:, 0], test_vectors_2d[:, 1], c=test_labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label="Task Index")
        plt.title("T-SNE Visualization of Test Descriptions")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid()
        plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cheetah_vel') # ['cheetah_dir', 'cheetah_vel', 'ant_dir', 'ML1-pick-place-v2']
    parser.add_argument('--dataset_mode', type=str, default='expert')
    parser.add_argument('--test_dataset_mode', type=str, default='expert')
    parser.add_argument('--train_prompt_mode', type=str, default='expert')
    parser.add_argument('--test_prompt_mode', type=str, default='expert')

    parser.add_argument('--prompt-episode', type=int, default=1)
    parser.add_argument('--prompt-length', type=int, default=5)
    parser.add_argument('--stochastic-prompt', action='store_true', default=True)
    parser.add_argument('--no-prompt', action='store_true', default=False)
    parser.add_argument('--no-r', action='store_true', default=False)
    parser.add_argument('--no-rtg', action='store_true', default=False)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--finetune_steps', type=int, default=10)
    parser.add_argument('--finetune_batch_size', type=int, default=256)
    parser.add_argument('--finetune_opt', action='store_true', default=True)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--no_state_normalize', action='store_true', default=False) 
    parser.add_argument('--average_state_mean', action='store_true', default=True) 
    parser.add_argument('--evaluation', action='store_true', default=False) 
    parser.add_argument('--render', action='store_true', default=False) 
    parser.add_argument('--load-path', type=str, default= None) # choose a model when in evaluation mode

    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000) # 10000*(number of environments)
    parser.add_argument('--num_eval_episodes', type=int, default=50) 
    parser.add_argument('--max_iters', type=int, default=5000) 
    parser.add_argument('--num_steps_per_iter', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--train_eval_interval', type=int, default=500)
    parser.add_argument('--test_eval_interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=500)

    args = parser.parse_args()
    experiment_mix_env('gym-experiment', variant=vars(args))