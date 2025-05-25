import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn as nn



# 定义 MOCO 模型
class MOCO(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, K=30):
        super(MOCO, self).__init__()
        self.K = K  # 队列大小

        # 查询编码器
        self.encoder_q = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # 键编码器
        self.encoder_k = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # 初始化键编码器的参数为查询编码器的参数
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 初始化队列
        self.register_buffer("queue", torch.randn(output_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self, m=0.999):
        """动量更新键编码器"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新队列"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # 队列大小必须是批次大小的整数倍

        # 替换队列中的旧键
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # 移动指针
        self.queue_ptr[0] = ptr

    def forward(self, x):
        return self.encoder_q(x)

def main():
    from read_bert import padded_final_nested_list  # 假设 test_list 是从 read_bert 模块导入的
    # 确保 test_list 是 NumPy 数组
    test_list = np.array(padded_final_nested_list)
    # test_list = np.array(final_nested_list)
    num_classes, num_samples_per_class, seq_len, hidden_dim = test_list.shape

    # 将测试数据展平为 (num_classes * num_samples_per_class, seq_len * hidden_dim)
    test_samples = test_list.reshape(-1, seq_len * hidden_dim)

    # 归一化测试数据
    min_vals = np.min(test_samples, axis=0)
    max_vals = np.max(test_samples, axis=0)
    mask = (min_vals == max_vals)
    max_vals[mask] += 1e-6  # 避免除以零
    test_samples = (test_samples - min_vals) / (max_vals - min_vals)

    # 为测试样本分配类别标签
    test_labels = np.repeat(np.arange(num_classes), num_samples_per_class)

    # 转换为 PyTorch 张量
    test_samples = torch.tensor(test_samples, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # 转换为 PyTorch 张量
    test_samples = torch.tensor(test_samples, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # 初始化 MOCO 模型
    input_dim = seq_len * hidden_dim  # 输入维度
    hidden_dim = 512  # 隐藏层维度
    output_dim = 256  # 输出维度
    model = MOCO(input_dim, hidden_dim, output_dim)

    # 加载训练好的模型参数
    model_path = "models/moco_300/moco_epoch_200.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式

    # 使用模型处理测试数据
    with torch.no_grad():
        test_embeddings = model(test_samples)

    # 将增强后的表征转换为 NumPy 数组
    test_embeddings = test_embeddings.numpy()

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=42)
    embedded_data = tsne.fit_transform(test_embeddings)

    # 可视化结果
    plt.figure(figsize=(10, 8))
    for label in np.unique(test_labels.numpy()):
        class_indices = np.where(test_labels.numpy() == label)
        plt.scatter(embedded_data[class_indices, 0], embedded_data[class_indices, 1],
                    label=f"Class {label}", alpha=0.6, s=10)

    plt.title("t-SNE Visualization of Test Task Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.show()

    labels = np.repeat(np.arange(5), 50)

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=3, random_state=42)  # 设置降维到 3 维
    tsne_result = tsne.fit_transform(test_embeddings)

    # 可视化结果
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], tsne_result[:, 2], c=labels, cmap="tab10", marker="o")
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    plt.title("t-SNE 3D Visualization")
    plt.show()
    #


if __name__ == "__main__":
    main()