import torch
from transformers import BertTokenizer, BertModel
import json
import  numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from moco_test import MOCO
n_max=1


def use_moco(descriptions):
    bert_model_path = './models/bert-base-uncased'  # bert路径

    # 从本地路径加载预训练的 BERT Tokenizer 和模型
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    bert_model = BertModel.from_pretrained(bert_model_path)
    input_dim = n_max * 768  # 输入维度
    hidden_dim = 512  # 隐藏层维度
    output_dim = 256  # 输出维度
    model = MOCO(input_dim, hidden_dim, output_dim)

    # 加载训练好的模型参数
    model_path = "models/moco_cheetah_dir_epoch_200.pth"  # moco模型路径
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    descriptions_embedding=[]
    for description in descriptions:
        encoded_input = tokenizer(description, return_tensors='pt', padding=True, truncation=True)
        # 将编码后的输入传递给模型
        with torch.no_grad():  # 不计算梯度
            output = bert_model(**encoded_input)

        # 获取最后一层的隐藏状态
        last_hidden_state = output.last_hidden_state[:,0,:].squeeze().detach().numpy()
        # print(last_hidden_state.shape)
        n = last_hidden_state.shape[0]
        #print('n:',n)
        # 创建一个 n_max * 768 的零矩阵
        padded_output = np.zeros((n_max, 768))
        # 将原始输出填充到零矩阵中
        padded_output[:n, :] = last_hidden_state
        #print('padded_output:',padded_output.shape)
        # 展平填充后的bert输出
        # print(padded_output.shape)
        padded_output = padded_output.flatten()
        #print('padded_output:',padded_output.shape)
        input_tensor = torch.tensor(padded_output, dtype=torch.float32)
        # 使用模型处理测试数据
        with torch.no_grad():
            description_embedding = model(input_tensor)

        # 将增强后的表征转换为 NumPy 数组
        description_embedding = description_embedding.numpy()
        descriptions_embedding.append(description_embedding)
        #print(description_embedding.shape)
    return descriptions_embedding





descriptions=["The cheetah should move forward at a speed of 0.75.","The cheetah should move forward at a speed of 0.75.","The cheetah should move forward at a speed of 0.75.",]
result=use_moco(descriptions)
print(len(result))
#
