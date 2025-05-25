import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel
import torch

# 指定本地模型和分词器的路径
model_path = './models/bert-base-uncased'

# 配置
data_folder = "data/cheetah_vel"
file_numbers = [2, 7, 15, 23, 26]
max_length = 128  # BERT的最大输入长度
num_clusters = 5  # 聚类数目
batch_size = 16   # 处理文本的批量大小

# 初始化BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

# 读取所有文本描述
all_texts = []
file_labels = []  # 记录每个文本描述来自哪个文件

for num in file_numbers:
    file_name = f"cheetah_vel-{num}-desc.py"
    file_path = os.path.join(data_folder, file_name)
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
            all_texts.extend(texts)
            file_labels.extend([num] * len(texts))
    else:
        print(f"文件 {file_path} 不存在，跳过")

# 将文本转换为BERT输入
def text_to_bert_embeddings(texts):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 分词
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                           max_length=max_length, return_tensors='pt')
        
        # 获取BERT输出
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        embeddings.extend(cls_embeddings)
    
    return np.array(embeddings)

# 获取所有文本的BERT嵌入
print("正在提取BERT特征...")
all_embeddings = text_to_bert_embeddings(all_texts)
print(f"提取完成，得到 {len(all_embeddings)} 个嵌入向量")

# 使用KMeans进行聚类
print("正在执行KMeans聚类...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(all_embeddings)
print("聚类完成")

# 将聚类中心与原始数据点合并
all_data_with_centers = np.vstack((all_embeddings, kmeans.cluster_centers_))

# 使用t-SNE降维以便可视化
print("正在进行t-SNE降维...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result_with_centers = tsne.fit_transform(all_data_with_centers)
print("降维完成")

# 分离降维后的数据点和聚类中心
tsne_result = tsne_result_with_centers[:len(all_embeddings)]
centers = tsne_result_with_centers[len(all_embeddings):]

# 可视化聚类结果
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'purple', 'orange']

for i, num in enumerate(file_numbers):
    file_indices = np.where(np.array(file_labels) == num)[0]
    plt.scatter(tsne_result[file_indices, 0], tsne_result[file_indices, 1], 
                c=colors[i], label=f"File {num}", alpha=0.6)

# 绘制聚类中心
for i in range(num_clusters):
    plt.scatter(centers[i, 0], centers[i, 1], c='black', marker='X', s=200, edgecolors='w')

plt.title('BERT Embeddings Clustering Visualization with t-SNE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.grid(True)
plt.show()