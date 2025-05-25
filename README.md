# 基于大语言模型的具身强化学习方法研究

BERT + 对比学习 + Decision Transformer


## Installation

Windows或Ubuntu中均可使用
 - 使用conda创建虚拟环境：

```
conda create --name prompt-dt python=3.8.5
conda activate prompt-dt
```
 - 注意，该项目基于Decision-Transformer，需要mujoco200版本 [mujoco-py repo](https://github.com/openai/mujoco-py).

 - 安装依赖项（部分依赖项需手动安装）：

```
# install dependencies
pip install -r requirements.txt

# install environments
./install_envs.sh
```
 - 我们使用 [wandb](https://wandb.ai/site?utm_source=google&utm_medium=cpc&utm_campaign=Performance-Max&utm_content=site&gclid=CjwKCAjwlqOXBhBqEiwA-hhitGcG5-wtdqoNgKyWdNpsRedsbEYyK9NeKcu8RFym6h8IatTjLFYliBoCbikQAvD_BwE) 进行可视化. 使用 [wandb quickstart doc](https://docs.wandb.ai/quickstart) 创建账户.

## Download Datasets
 - 数据在此 [Google Drive link](https://drive.google.com/drive/folders/1six767uD8yfdgoGIYW86sJY-fmMdYq7e?usp=sharing).
 - Download the "data" folder.

```
wget -O data.zip 'https://drive.google.com/uc?export=download&id=1rZufm-XRq1Ig-56DejkQUX1si_WzCGBe&confirm=True' 
unzip data.zip
rm data.zip
```
 - 文件夹格式如下.
```
.
├── config
├── data
│   ├── ant_dir
│   ├── cheetah_dir
│   ├── cheetah_vel
│   └── ML1-pick-place-v2
├── dataset
├── envs
├── prompt_dt
├── models
└── ...
```
## Run Experiments
```
# Prompt-DT
python pdt_main.py --env cheetah_dir # choices:['cheetah_dir', 'cheetah_vel', 'ant_dir', 'ML1-pick-place-v2']

# Prompt-MT-BC
python pdt_main.py --no-rtg --no-r

# MT-ORL
python pdt_main.py --no-prompt

# MT-BC-Finetune
python pdt_main.py --no-prompt --no-rtg --no-r --finetune
```

## Attention!
共有4种文本处理方式，根据需要选择
- desc2arrray 
- desc2arrrayCLS
- load_descriptions_new
- desc2arrrayMOCO
train_context_encoder.py是基于世界模型的轨迹表征处理函数

## Acknowledgements
The code for prompt-dt is based on [decision-transformer](https://github.com/kzl/decision-transformer). We build environments based on repos including [macaw](https://github.com/eric-mitchell/macaw), [rand_param_envs](https://github.com/dennisl88/rand_param_envs), and [metaworld](https://github.com/rlworkgroup/metaworld).

