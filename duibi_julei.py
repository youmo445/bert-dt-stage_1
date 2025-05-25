import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# 配置参数
class Config:
    batch_size = 256
    hidden_dim = 128
    projection_dim = 64
    temperature = 0.1
    learning_rate = 3e-4
    epochs = 20
    max_length = 128
    vocab_size = 10000  # 假设的词汇表大小

config = Config()

# 数据增强模块
class TextAugmenter:
    def __init__(self, augmentation_rate=0.2):
        self.aug_rate = augmentation_rate
        
    def __call__(self, text):
        return self.random_deletion(text)
    
    def random_deletion(self, text):
        words = text.split()
        if len(words) < 2:
            return text
        # 随机删除部分词汇
        keep_prob = 1 - self.aug_rate
        new_words = [word for word in words if random.random() < keep_prob]
        return ' '.join(new_words) if len(new_words) > 0 else text

# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, texts, augmenter):
        self.texts = texts
        self.augmenter = augmenter
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        aug1 = self.augmenter(text)
        aug2 = self.augmenter(text)
        return aug1, aug2

# 文本编码器
class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=8,
                dim_feedforward=256
            ),
            num_layers=3
        )
        self.pooler = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, input_ids):
        # 输入处理（简化的tokenization）
        embedded = self.embedding(input_ids)
        features = self.transformer(embedded)
        pooled = self.pooler(features.permute(0,2,1)).squeeze()
        return pooled

# 对比学习模型
class ContrastiveModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TextEncoder(config)
        self.projection_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.projection_dim),
            nn.ReLU(),
            nn.Linear(config.projection_dim, config.projection_dim)
        )
        
    def forward(self, input_ids):
        features = self.encoder(input_ids)
        projections = self.projection_head(features)
        return F.normalize(projections, p=2, dim=1)

# NT-Xent损失函数
class NTXentLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        
        # 合并特征
        features = torch.cat([z_i, z_j], dim=0)
        
        # 计算相似度矩阵
        similarity_matrix = F.cosine_similarity(
            features.unsqueeze(1), features.unsqueeze(0), dim=2
        ) / self.temperature
        
        # 创建标签
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size), 
            torch.arange(batch_size)
        ], dim=0).to(z_i.device)
        
        # 排除自身对比
        mask = torch.eye(2*batch_size, dtype=torch.bool).to(z_i.device)
        similarity_matrix = similarity_matrix[~mask].view(2*batch_size, -1)
        
        return self.cross_entropy(similarity_matrix, labels)

# 评估器
class Evaluator:
    @staticmethod
    def calculate_similarity_matrix(embeddings):
        return F.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )
    
    @staticmethod
    def intra_inter_similarity(matrix, labels):
        intra_sim = []
        inter_sim = []
        for i in range(len(matrix)):
            for j in range(i+1, len(matrix)):
                if labels[i] == labels[j]:
                    intra_sim.append(matrix[i,j].item())
                else:
                    inter_sim.append(matrix[i,j].item())
        return np.mean(intra_sim), np.mean(inter_sim)

# 训练流程
def train():
    # 示例数据（替换为真实数据）
    texts = [
        "This is a document about machine learning",
        "Deep learning requires powerful hardware",
        "The weather is sunny today",
        "Climate change affects global temperature",
        # ... 添加更多样本
    ] * 100  # 扩展样本量
    
    # 初始化组件
    augmenter = TextAugmenter(augmentation_rate=0.2)
    dataset = TextDataset(texts, augmenter)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    model = ContrastiveModel(config)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = NTXentLoss(config.temperature)
    
    # 训练循环
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            # 伪tokenization（替换为真实tokenizer）
            aug1, aug2 = batch
            input_ids1 = torch.randint(0, config.vocab_size, (len(aug1), config.max_length))
            input_ids2 = torch.randint(0, config.vocab_size, (len(aug2), config.max_length))
            
            # 前向传播
            proj1 = model(input_ids1)
            proj2 = model(input_ids2)
            
            # 计算损失
            loss = criterion(proj1, proj2)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.4f}")
        
        # 评估相似度
        if (epoch+1) % 5 == 0:
            test_samples = [
                "Machine learning algorithms",
                "ML requires computational power",
                "Today's weather forecast",
                "Global warming impacts"
            ]
            # 生成嵌入
            with torch.no_grad():
                inputs = torch.randint(0, config.vocab_size, (len(test_samples), config.max_length))
                embeddings = model(inputs)
            
            sim_matrix = Evaluator.calculate_similarity_matrix(embeddings)
            intra_sim, inter_sim = Evaluator.intra_inter_similarity(
                sim_matrix, labels=[0,0,1,1]
            )
            print(f"Intra-class similarity: {intra_sim:.3f}")
            print(f"Inter-class similarity: {inter_sim:.3f}")
            print("----------------------")

if __name__ == "__main__":
    torch.manual_seed(42)
    train()