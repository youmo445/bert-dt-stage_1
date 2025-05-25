from ast import parse
import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys
import time
import itertools

from prompt_dt.prompt_decision_transformer import PromptDecisionTransformer
from prompt_dt.prompt_seq_trainer import PromptSequenceTrainer
from prompt_dt.prompt_utils import get_env_list
from prompt_dt.prompt_utils import get_prompt_batch, get_prompt, get_batch, get_batch_finetune
from prompt_dt.prompt_utils import process_total_data_mean, load_data_prompt, process_info
from prompt_dt.prompt_utils import eval_episodes

from collections import namedtuple
import json, pickle, os

def evaluate_model(model_path, env_name, variant):
    # 构造测试环境
    cur_dir = os.getcwd()
    config_save_path = os.path.join(cur_dir, 'config')
    data_save_path = os.path.join(cur_dir, 'data')

    # 加载测试环境配置
    config_path_dict = {
        'cheetah_vel': "cheetah_vel/cheetah_vel_40.json",
        'cheetah_dir': "cheetah_dir/cheetah_dir_2.json",
        'ant_dir': "ant_dir/ant_dir_50.json",
        'ML1-pick-place-v2': "ML1-pick-place-v2/ML1_pick_place.json",
    }
    task_config_path = os.path.join(config_save_path, config_path_dict[env_name])
    with open(task_config_path, 'r') as f:
        task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    # 构造测试环境列表
    test_env_name_list = [f"{env_name}-{task_id}" for task_id in task_config.test_tasks]
    _, test_env_list = get_env_list(test_env_name_list, config_save_path, variant['device'])

    # 加载测试数据集
    test_dataset_mode = variant['test_dataset_mode']
    test_prompt_mode = variant['test_prompt_mode']
    test_trajectories_list, test_prompt_trajectories_list = load_data_prompt(
        test_env_name_list, data_save_path, test_dataset_mode, test_prompt_mode, variant
    )

    # 加载模型
    state_dim = test_env_list[0].observation_space.shape[0]
    act_dim = test_env_list[0].action_space.shape[0]

    model = PromptDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=variant['K'],
        max_ep_len=1000,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )
    model.load_state_dict(torch.load(model_path, map_location=variant['device']))
    model.to(variant['device'])

    # 构造 Trainer（仅用于评估）
    trainer = PromptSequenceTrainer(
        model=model,
        optimizer=None,
        batch_size=variant['batch_size'],
        get_batch=None,
        scheduler=None,
        loss_fn=None,
        eval_fns=None,
        get_prompt=None,
        get_prompt_batch=None
    )

    # 评估模型
    print(f"Evaluating model from {model_path}...")
    eval_logs = trainer.eval_iteration_multienv(
        get_prompt=None,
        prompt_trajectories_list=test_prompt_trajectories_list,
        eval_episodes=eval_episodes,
        env_name_list=test_env_name_list,
        info=process_info(test_env_name_list, test_trajectories_list, {}, variant['mode'], test_dataset_mode, 1.0, variant),
        variant=variant,
        env_list=test_env_list,
        iter_num=0,
        print_logs=True,
        no_prompt=variant['no_prompt'],
        group='eval'
    )
    print("Evaluation Results:", eval_logs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained model.")
    parser.add_argument('--env', type=str, default='cheetah_vel', help="Environment name.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument('--K', type=int, default=20, help="Sequence length.")
    parser.add_argument('--embed_dim', type=int, default=128, help="Embedding dimension.")
    parser.add_argument('--n_layer', type=int, default=3, help="Number of layers.")
    parser.add_argument('--n_head', type=int, default=1, help="Number of attention heads.")
    parser.add_argument('--activation_function', type=str, default='relu', help="Activation function.")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use (cuda or cpu).")
    parser.add_argument('--no-prompt', action='store_true', default=False, help="Disable prompt usage.")
    parser.add_argument('--test-dataset-mode', type=str, default='expert', help="Test dataset mode.")
    parser.add_argument('--test-prompt-mode', type=str, default='expert', help="Test prompt mode.")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size for evaluation.")
    args = parser.parse_args()

    variant = vars(args)
    evaluate_model(args.model_path, args.env, variant)