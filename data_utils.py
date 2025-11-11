import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  # 使用新的导入路径
from sklearn.model_selection import train_test_split
from config import config


def create_improved_synthetic_dataset(num_samples=None):
    """创建更真实的合成数据集"""
    if num_samples is None:
        num_samples = config.NUM_SAMPLES

    data_list = []

    # 为每个类别创建更有区分度的特征
    class_centers = {
        0: {'mri': [0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.9, 0.8, 0.7, 0.8],  # 正常
            'cog': [28, 30, 25, 29, 27, 26, 30, 28, 29, 27]},
        1: {'mri': [0.5, 0.4, 0.6, 0.3, 0.5, 0.4, 0.6, 0.5, 0.4, 0.5],  # MCI
            'cog': [22, 24, 20, 23, 21, 20, 24, 22, 23, 21]},
        2: {'mri': [0.2, 0.3, 0.1, 0.4, 0.2, 0.3, 0.1, 0.2, 0.3, 0.2],  # AD
            'cog': [15, 18, 12, 17, 14, 13, 18, 15, 17, 14]}
    }

    for i in range(num_samples):
        # 根据标签生成更有区分度的特征
        label = generate_label(i, num_samples).item()
        center = class_centers[label]

        # 模拟图结构数据 - 基于类别调整连接概率
        num_nodes = 20
        base_prob = 0.3
        if label == 0:  # 正常 - 更多连接
            conn_prob = base_prob + 0.2
        elif label == 1:  # MCI - 中等连接
            conn_prob = base_prob
        else:  # AD - 较少连接
            conn_prob = base_prob - 0.1

        node_features = torch.randn(num_nodes, config.NODE_FEATURE_DIM)
        edge_index = create_brain_network_with_prob(num_nodes, conn_prob)

        # 生成基于类别中心的特征（添加噪声）
        mri_features = torch.tensor(center['mri'], dtype=torch.float) + torch.randn(10) * 0.1
        cog_features = torch.tensor(center['cog'][:8], dtype=torch.float) + torch.randn(8) * 2
        clin_features = torch.randn(6)  # 临床特征
        genetic_features = torch.randn(4)  # 遗传特征

        # 确保特征在合理范围内
        mri_features = torch.sigmoid(mri_features).unsqueeze(0)
        cog_features = (cog_features / 30.0).unsqueeze(0)  # 归一化
        clin_features = torch.sigmoid(clin_features).unsqueeze(0)
        genetic_features = torch.sigmoid(genetic_features).unsqueeze(0)

        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.tensor([label]),
            mri_features=mri_features,
            cog_features=cog_features,
            clin_features=clin_features,
            genetic_features=genetic_features
        )

        data_list.append(graph_data)

    return data_list


def create_brain_network_with_prob(num_nodes, connection_prob):
    """根据概率创建脑网络连接"""
    edge_index = []
    for j in range(num_nodes):
        for k in range(j + 1, num_nodes):
            if np.random.random() < connection_prob:
                edge_index.append([j, k])
                edge_index.append([k, j])

    if len(edge_index) == 0:
        edge_index = [[0, 1], [1, 0], [0, 2], [2, 0]]  # 确保基本连接

    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


def generate_label(sample_idx, total_samples):
    """生成样本标签"""
    if sample_idx < total_samples * 0.4:
        return torch.tensor([0])  # 正常
    elif sample_idx < total_samples * 0.7:
        return torch.tensor([1])  # MCI
    else:
        return torch.tensor([2])  # AD


def prepare_data_loaders(dataset):
    """准备数据加载器"""
    train_data, test_data = train_test_split(
        dataset, test_size=config.TRAIN_TEST_SPLIT, random_state=42
    )
    train_data, val_data = train_test_split(
        train_data, test_size=config.TRAIN_VAL_SPLIT, random_state=42
    )

    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_data, val_data, test_data


def get_dataset_info(dataset, train_data, val_data, test_data):
    """获取数据集信息"""
    info = {
        'total_samples': len(dataset),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'class_distribution': {
            'Normal': sum(1 for data in dataset if data.y.item() == 0),
            'MCI': sum(1 for data in dataset if data.y.item() == 1),
            'AD': sum(1 for data in dataset if data.y.item() == 2)
        }
    }
    return info