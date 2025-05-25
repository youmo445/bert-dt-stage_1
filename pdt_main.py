from ast import parse
import sys
sys.path.append('envs/mujoco-control-envs/mujoco_control_envs/tp_envs/rand_param_envs/rand_param_envs/gym')
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
from prompt_dt.prompt_utils import get_env_list,desc2arraynew
from prompt_dt.prompt_utils import get_prompt_batch, get_prompt, get_batch, get_batch_finetune,get_descriptions, get_descriptions_batch
from prompt_dt.prompt_utils import process_total_data_mean, load_data_prompt, process_info, load_descriptions, desc2arrray, desc2arrrayCLS, load_descriptions_new, desc2arrrayMOCO
from prompt_dt.prompt_utils import eval_episodes

from collections import namedtuple
import json, pickle, os

def experiment_mix_env(
        exp_prefix,
        variant,
):
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
    print('test_prompt_traj_list',np.array(test_prompt_trajectories_list).shape)#(2,5)
    train_description_list=load_descriptions_new(train_env_name_list, data_save_path, desc_mode='train',args=args)
    test_description_list=load_descriptions_new(test_env_name_list, data_save_path, desc_mode='test',args=args)
    

    '''
    注意，如果是moco的话把下面的desc2arraynew换成desc2arrrayMOCO
    '''
    train_description_list=desc2arraynew(train_description_list)#(2,40,n,768)/(2,40,1,768)
    train_description_list=np.array(train_description_list)
    print('train_description_list',train_description_list.shape)#(2,40,n,768)
    test_description_list=desc2arraynew(test_description_list)#(2,10,n,768)
    test_description_list=np.array(test_description_list)
    print('test_description_list',test_description_list.shape)#(2,10,n,768)
    # 将 train_description_list 从 (35, 40, 128) 转换为 (35, 40, 1, 128)
    train_description_list = np.expand_dims(train_description_list, axis=2)  # 在第 2 维添加一个维度
    print('train_description_list', train_description_list.shape)  # (35, 40, 1, 128)

    # 将 test_description_list 从 (35, 10, 128) 转换为 (35, 10, 1, 128)
    test_description_list = np.expand_dims(test_description_list, axis=2)  # 在第 2 维添加一个维度
    print('test_description_list', test_description_list.shape)  # (35, 10, 1, 128)
    '''
    如果是用moco的话不用向上面一样添加维度
    '''

    # change to total train trajecotry 
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
    # process test info
    test_info = process_info(test_env_name_list, test_trajectories_list, test_info, mode, test_dataset_mode, pct_traj, variant)
    print('ending process train and test datasets-----------------')
    ######
    # construct dt model and trainer
    ######
    
    exp_prefix = exp_prefix + '-' + args.env
    num_env = len(train_env_name_list)
    group_name = f'{exp_prefix}-{str(num_env)}-Env-{dataset_mode}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    state_dim = test_env_list[0].observation_space.shape[0]
    act_dim = test_env_list[0].action_space.shape[0]

    model = PromptDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
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
    model = model.to(device=device)
    print('ending constructing dt model-----------------')
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    env_name = train_env_name_list[0]
    print('train_des',len(train_description_list))
    trainer = PromptSequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,# 16
        get_batch=get_batch(trajectories_list[0], info[env_name], variant),#将get_batch函数里的fn传给新get_batch
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),# MSE
        eval_fns=None,
        get_descriptions=get_descriptions(train_description_list[0], info[env_name], variant),#将get_pr函数里的fn传给新get_batch
        get_descriptions_batch=get_descriptions_batch(trajectories_list, train_description_list, info, variant, train_env_name_list)#将get_pr_b函数里的fn传给新get_batch
    )
    # 这里会跑enter-prompt-utils_get_batch
    # enter-prompt-utils_get_prompt
    # enter-prompt-utils_get_prompt_batch
    print('ending constructing dt model and trainer---------')

    if not variant['evaluation']:
        ######
        # start training
        ######
        print('start training')
        if log_to_wandb:
            wandb.init(
                name=exp_prefix,
                group=group_name,
                project='prompt-decision-transformer',
                config=variant
            )
            save_path += wandb.run.name
            os.mkdir(save_path)

        # construct model post fix
        model_post_fix = '_TRAIN_'+variant['train_prompt_mode']+'_TEST_'+variant['test_prompt_mode']
        if variant['no_prompt']:
            model_post_fix += '_NO_PROMPT'
        if variant['finetune']:
            model_post_fix += '_FINETUNE'
        if variant['no_r']:
            model_post_fix += '_NO_R'
        print('进入5000轮迭代')
        for iter in range(variant['max_iters']):#一共2个环境。iter跑5000轮
            
            env_id = iter % num_env
            env_name = train_env_name_list[env_id]#间隔训练两个环境
            outputs = trainer.pure_train_iteration_mix(
                num_steps=variant['num_steps_per_iter'], #10
                no_prompt=args.no_prompt
                )

            # start evaluation
            if iter % args.test_eval_interval == 0:#满100轮
                #input('满100轮')
                # evaluate test
                if not args.finetune:
                    test_eval_logs = trainer.eval_iteration_multienv(
                        get_descriptions, test_description_list,
                        eval_episodes, test_env_name_list, test_info, variant, test_env_list, iter_num=iter + 1, 
                        print_logs=True, no_prompt=args.no_prompt, group='test')
                    outputs.update(test_eval_logs)
                else:
                    test_eval_logs = trainer.finetune_eval_iteration_multienv(
                        get_descriptions, get_batch_finetune, test_description_list, test_trajectories_list,
                        eval_episodes, test_env_name_list, test_info, 
                        variant, test_env_list, iter_num=iter + 1, 
                        print_logs=True, no_prompt=args.no_prompt, 
                        group='finetune-test', finetune_opt=variant['finetune_opt'])
                    outputs.update(test_eval_logs)
            
            if iter % args.train_eval_interval == 0:#满500轮
                #input('满500轮')
                # evaluate train
                train_eval_logs = trainer.eval_iteration_multienv(
                    get_descriptions, train_description_list,
                    eval_episodes, train_env_name_list, info, variant, env_list, iter_num=iter + 1, 
                    print_logs=True, no_prompt=args.no_prompt, group='train')
                outputs.update(train_eval_logs)

            if iter % variant['save_interval'] == 0:
                trainer.save_model(
                    env_name=args.env, 
                    postfix=model_post_fix+'_iter_'+str(iter), 
                    folder=save_path)

            outputs.update({"global_step": iter}) # set global step as iteration

            if log_to_wandb:
                wandb.log(outputs)
        
        trainer.save_model(env_name=args.env,  postfix=model_post_fix+'_iter_'+str(iter),  folder=save_path)

    else:
        ####
        # start evaluating
        ####
        print('start evaluating')
        saved_model_path = os.path.join(save_path, variant['load_path'])
        model.load_state_dict(torch.load(saved_model_path))
        print('model initialized from: ', saved_model_path)
        eval_iter_num = int(saved_model_path.split('_')[-1])

        eval_logs = trainer.eval_iteration_multienv(
                    get_descriptions, test_description_list,
                    eval_episodes, test_env_name_list, test_info, variant, test_env_list, iter_num=eval_iter_num, 
                    print_logs=True, no_prompt=args.no_prompt, group='eval')

        
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
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    parser.add_argument('--train_eval_interval', type=int, default=500)
    parser.add_argument('--test_eval_interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=500)

    args = parser.parse_args()
    experiment_mix_env('gym-experiment', variant=vars(args))