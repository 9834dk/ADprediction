import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from config import config


class ImprovedMultimodalGNN(nn.Module):
    def __init__(self):
        super(ImprovedMultimodalGNN, self).__init__()

        # 图卷积层
        self.conv1 = GCNConv(config.NODE_FEATURE_DIM, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

        # 多模态特征融合 - 统一输出维度
        self.mri_fc = nn.Linear(config.MRI_DIM, 16)
        self.cog_fc = nn.Linear(config.COG_DIM, 16)
        self.clin_fc = nn.Linear(config.CLIN_DIM, 16)
        self.genetic_fc = nn.Linear(config.GENETIC_DIM, 16)

        # 计算分类器输入维度
        # 图特征: 16 (mean) + 16 (max) = 32
        # 多模态特征: 16 * 4 = 64
        # 总计: 32 + 64 = 96
        classifier_input_dim = 32 + 64  # 图特征 + 多模态特征

        # 改进的分类器
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, config.NUM_CLASSES)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GNN处理
        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index)))
        graph_features = F.relu(self.conv3(x2, edge_index))

        # 双池化策略
        graph_mean = global_mean_pool(graph_features, batch)
        graph_max = global_max_pool(graph_features, batch)
        graph_embedding = torch.cat([graph_mean, graph_max], dim=1)  # 32维

        # 多模态特征处理
        mri_features = F.relu(self.mri_fc(data.mri_features))
        cog_features = F.relu(self.cog_fc(data.cog_features))
        clin_features = F.relu(self.clin_fc(data.clin_features))
        genetic_features = F.relu(self.genetic_fc(data.genetic_features))

        # 特征拼接
        multimodal_features = torch.cat([
            mri_features, cog_features, clin_features, genetic_features
        ], dim=1)  # 64维

        # 最终特征融合
        combined_features = torch.cat([graph_embedding, multimodal_features], dim=1)  # 96维

        # 分类
        output = self.classifier(combined_features)

        return F.log_softmax(output, dim=1)


# 简化版本，确保能运行
class SimpleMultimodalGNN(nn.Module):
    def __init__(self):
        super(SimpleMultimodalGNN, self).__init__()

        # 简化的图卷积层
        self.conv1 = GCNConv(config.NODE_FEATURE_DIM, 32)
        self.conv2 = GCNConv(32, 16)

        # 多模态特征融合
        self.mri_fc = nn.Linear(config.MRI_DIM, 8)
        self.cog_fc = nn.Linear(config.COG_DIM, 8)
        self.clin_fc = nn.Linear(config.CLIN_DIM, 8)
        self.genetic_fc = nn.Linear(config.GENETIC_DIM, 8)

        # 计算分类器输入维度
        # 图特征: 16 + 多模态特征: 32 = 48
        classifier_input_dim = 16 + 32

        # 简化分类器
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, config.NUM_CLASSES)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GNN处理
        x = F.relu(self.conv1(x, edge_index))
        graph_features = F.relu(self.conv2(x, edge_index))

        # 图级表示
        graph_embedding = global_mean_pool(graph_features, batch)  # 16维

        # 多模态特征处理
        mri_features = F.relu(self.mri_fc(data.mri_features))
        cog_features = F.relu(self.cog_fc(data.cog_features))
        clin_features = F.relu(self.clin_fc(data.clin_features))
        genetic_features = F.relu(self.genetic_fc(data.genetic_features))

        # 特征拼接
        multimodal_features = torch.cat([
            mri_features, cog_features, clin_features, genetic_features
        ], dim=1)  # 32维

        # 最终特征融合
        combined_features = torch.cat([graph_embedding, multimodal_features], dim=1)  # 48维

        # 分类
        output = self.classifier(combined_features)

        return F.log_softmax(output, dim=1)


# 最简版本，确保一定能运行
class BasicMultimodalGNN(nn.Module):
    def __init__(self):
        super(BasicMultimodalGNN, self).__init__()

        # 基本的图卷积层
        self.conv1 = GCNConv(config.NODE_FEATURE_DIM, 16)

        # 多模态特征融合
        self.mri_fc = nn.Linear(config.MRI_DIM, 4)
        self.cog_fc = nn.Linear(config.COG_DIM, 4)
        self.clin_fc = nn.Linear(config.CLIN_DIM, 4)
        self.genetic_fc = nn.Linear(config.GENETIC_DIM, 4)

        # 计算分类器输入维度
        # 图特征: 16 + 多模态特征: 16 = 32
        classifier_input_dim = 16 + 16

        # 基本分类器
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, config.NUM_CLASSES)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GNN处理
        graph_features = F.relu(self.conv1(x, edge_index))

        # 图级表示
        graph_embedding = global_mean_pool(graph_features, batch)  # 16维

        # 多模态特征处理
        mri_features = F.relu(self.mri_fc(data.mri_features))
        cog_features = F.relu(self.cog_fc(data.cog_features))
        clin_features = F.relu(self.clin_fc(data.clin_features))
        genetic_features = F.relu(self.genetic_fc(data.genetic_features))

        # 特征拼接
        multimodal_features = torch.cat([
            mri_features, cog_features, clin_features, genetic_features
        ], dim=1)  # 16维

        # 最终特征融合
        combined_features = torch.cat([graph_embedding, multimodal_features], dim=1)  # 32维

        # 分类
        output = self.classifier(combined_features)

        return F.log_softmax(output, dim=1)