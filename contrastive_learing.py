import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
# 指定本地模型和分词器的路径
model_path = './models/bert-base-uncased'

# 配置
data_folder = "data/cheetah_vel"
max_length = 128  # BERT的最大输入长度
batch_size = 16   # 处理文本的批量大小
learning_rate = 1e-3
epochs = 100

# 初始化BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model = BertModel.from_pretrained(model_path)

# 定义降维模型
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.normalize = nn.functional.normalize

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = self.normalize(x, dim=-1)
        return x

# 定义对比损失函数（NT-Xent Loss）
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        # 计算相似性矩阵
        sim_matrix = self.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0)) / self.temperature
        labels = torch.arange(z_i.size(0)).to(z_i.device)
        loss = nn.CrossEntropyLoss()(sim_matrix, labels)
        return loss

# 读取所有文本描述
def load_descriptions(data_folder):
    descriptions_list = []
    task_labels = []  # 记录每个描述属于哪个任务
    task_id = 0

    for file_name in os.listdir(data_folder):
        if file_name.endswith("-desc.py"):
            file_path = os.path.join(data_folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
                descriptions_list.extend(texts)
                task_labels.extend([task_id] * len(texts))
            task_id += 1

    return descriptions_list, task_labels

# 将文本转换为BERT嵌入
def text_to_bert_embeddings(texts):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 分词
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                           max_length=max_length, return_tensors='pt')
        
        # 获取BERT输出
        with torch.no_grad():
            outputs = bert_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        embeddings.append(cls_embeddings)
    
    return torch.cat(embeddings, dim=0)  # (num_texts, 768)

# 加载描述数据
descriptions_list, task_labels = load_descriptions(data_folder)
task_labels = torch.tensor(task_labels)
print(f"加载了 {len(descriptions_list)} 条描述，属于 {len(set(task_labels.tolist()))} 个任务")

# 获取所有文本的BERT嵌入
print("正在提取BERT特征...")
all_embeddings = text_to_bert_embeddings(descriptions_list)
print(f"提取完成，得到 {all_embeddings.shape[0]} 个嵌入向量")

# 初始化降维模型和优化器
projection_head = ProjectionHead()
optimizer = optim.Adam(projection_head.parameters(), lr=learning_rate)
criterion = NTXentLoss()

# 训练对比学习模型
def train_contrastive_learning(embeddings, labels, epochs=100):
    for epoch in range(epochs):
        # 打乱数据
        perm = torch.randperm(embeddings.size(0))
        embeddings = embeddings[perm]
        labels = labels[perm]

        # 投影到128维
        projected_embeddings = projection_head(embeddings)

        # 计算对比损失
        loss = criterion(projected_embeddings, projected_embeddings)

        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# 开始训练
train_contrastive_learning(all_embeddings, task_labels, epochs=epochs)

# 使用 t-SNE 对嵌入向量降维
print("正在使用 t-SNE 降维...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
reduced_embeddings = tsne.fit_transform(all_embeddings.numpy())
print("t-SNE 降维完成")

# 可视化降维结果
def visualize_embeddings(embeddings, labels):
    plt.figure(figsize=(12, 8))
    num_tasks = len(set(labels.tolist()))
    palette = sns.color_palette("hsv", num_tasks)  # 为每个任务分配不同颜色

    for task_id in range(num_tasks):
        task_indices = (labels == task_id).nonzero(as_tuple=True)[0]
        plt.scatter(
            embeddings[task_indices, 0],
            embeddings[task_indices, 1],
            label=f"Task {task_id}",
            alpha=0.6,
            color=palette[task_id]
        )

    plt.title("t-SNE Visualization of Task Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()

# 调用可视化函数
visualize_embeddings(reduced_embeddings, task_labels)