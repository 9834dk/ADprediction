import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.nn import AttentionalAggregation
from config import config


class GraphAttentionLayer(nn.Module):
    """图注意力层"""

    def __init__(self, in_features, out_features, dropout=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

    def forward(self, x, edge_index):
        # x: [num_nodes, in_features]
        # edge_index: [2, num_edges]

        h = self.W(x)  # [num_nodes, out_features]

        # 计算注意力系数
        row, col = edge_index
        h_cat = torch.cat([h[row], h[col]], dim=1)  # [num_edges, 2 * out_features]
        e = self.leakyrelu(self.a(h_cat)).squeeze()  # [num_edges]

        # 应用softmax归一化
        attention = torch.zeros(x.size(0), x.size(0), device=x.device)
        attention[row, col] = e
        attention = F.softmax(attention, dim=1)

        # 应用注意力机制
        h_prime = torch.matmul(attention, h)

        return F.elu(h_prime)


class MultiModalAttention(nn.Module):
    """多模态注意力融合模块"""

    def __init__(self, feature_dims, hidden_dim=32):
        super(MultiModalAttention, self).__init__()
        self.num_modalities = len(feature_dims)

        # 为每个模态创建转换层
        self.transform_layers = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in feature_dims
        ])

        # 注意力权重计算
        self.attention_weights = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, modalities):
        # modalities: 模态特征列表 [mri, cog, clin, genetic]

        # 转换每个模态到相同维度
        transformed = []
        for i, modality in enumerate(modalities):
            transformed.append(F.relu(self.transform_layers[i](modality)))

        # 计算注意力权重
        attention_scores = []
        for modality in transformed:
            score = self.attention_weights(modality)
            attention_scores.append(score)

        # 拼接并应用softmax
        stacked_scores = torch.stack(attention_scores, dim=1)  # [batch, num_modalities, 1]
        attention_weights = F.softmax(stacked_scores, dim=1)  # [batch, num_modalities, 1]

        # 加权融合
        stacked_modalities = torch.stack(transformed, dim=1)  # [batch, num_modalities, hidden_dim]
        fused_features = torch.sum(attention_weights * stacked_modalities, dim=1)

        return fused_features, attention_weights.squeeze(-1)


class ImprovedMultimodalGNN(nn.Module):
    def __init__(self):
        super(ImprovedMultimodalGNN, self).__init__()

        # 图卷积层 + 图注意力
        self.conv1 = GCNConv(config.NODE_FEATURE_DIM, 64)
        self.graph_attention1 = GraphAttentionLayer(64, 64)
        self.conv2 = GCNConv(64, 32)
        self.graph_attention2 = GraphAttentionLayer(32, 32)
        self.conv3 = GCNConv(32, 16)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

        # 全局图注意力池化
        self.global_attention = AttentionalAggregation(
            nn.Sequential(
                nn.Linear(16, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )
        )

        # 多模态特征融合 - 统一输出维度
        self.mri_fc = nn.Linear(config.MRI_DIM, 16)
        self.cog_fc = nn.Linear(config.COG_DIM, 16)
        self.clin_fc = nn.Linear(config.CLIN_DIM, 16)
        self.genetic_fc = nn.Linear(config.GENETIC_DIM, 16)

        # 多模态注意力融合
        modal_dims = [16, 16, 16, 16]  # 每个模态的特征维度
        self.multimodal_attention = MultiModalAttention(modal_dims, hidden_dim=16)

        # 计算分类器输入维度
        # 图特征: 16 (attention pooling) + 多模态特征: 16 = 32
        classifier_input_dim = 16 + 16

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

        # GNN处理 + 图注意力
        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1_att = self.graph_attention1(x1, edge_index)
        x1_combined = x1 + x1_att  # 残差连接

        x2 = F.relu(self.bn2(self.conv2(x1_combined, edge_index)))
        x2_att = self.graph_attention2(x2, edge_index)
        x2_combined = x2 + x2_att  # 残差连接

        graph_features = F.relu(self.conv3(x2_combined, edge_index))

        # 全局注意力池化
        graph_embedding = self.global_attention(graph_features, batch)  # 16维

        # 多模态特征处理
        mri_features = F.relu(self.mri_fc(data.mri_features))
        cog_features = F.relu(self.cog_fc(data.cog_features))
        clin_features = F.relu(self.clin_fc(data.clin_features))
        genetic_features = F.relu(self.genetic_fc(data.genetic_features))

        # 多模态注意力融合
        modalities = [mri_features, cog_features, clin_features, genetic_features]
        multimodal_features, attention_weights = self.multimodal_attention(modalities)  # 16维

        # 最终特征融合
        combined_features = torch.cat([graph_embedding, multimodal_features], dim=1)  # 32维

        # 分类
        output = self.classifier(combined_features)

        return F.log_softmax(output, dim=1), attention_weights


# 简化版本，确保能运行
class SimpleMultimodalGNN(nn.Module):
    def __init__(self):
        super(SimpleMultimodalGNN, self).__init__()

        # 简化的图卷积层
        self.conv1 = GCNConv(config.NODE_FEATURE_DIM, 32)
        self.conv2 = GCNConv(32, 16)

        # 全局注意力池化
        self.global_attention = GlobalAttention(
            nn.Sequential(
                nn.Linear(16, 16),
                nn.Tanh(),
                nn.Linear(16, 1)
            )
        )

        # 多模态特征融合
        self.mri_fc = nn.Linear(config.MRI_DIM, 8)
        self.cog_fc = nn.Linear(config.COG_DIM, 8)
        self.clin_fc = nn.Linear(config.CLIN_DIM, 8)
        self.genetic_fc = nn.Linear(config.GENETIC_DIM, 8)

        # 多模态注意力融合
        modal_dims = [8, 8, 8, 8]
        self.multimodal_attention = MultiModalAttention(modal_dims, hidden_dim=8)

        # 计算分类器输入维度
        # 图特征: 16 + 多模态特征: 8 = 24
        classifier_input_dim = 16 + 8

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

        # 图级表示 - 使用注意力池化
        graph_embedding = self.global_attention(graph_features, batch)  # 16维

        # 多模态特征处理
        mri_features = F.relu(self.mri_fc(data.mri_features))
        cog_features = F.relu(self.cog_fc(data.cog_features))
        clin_features = F.relu(self.clin_fc(data.clin_features))
        genetic_features = F.relu(self.genetic_fc(data.genetic_features))

        # 多模态注意力融合
        modalities = [mri_features, cog_features, clin_features, genetic_features]
        multimodal_features, attention_weights = self.multimodal_attention(modalities)  # 8维

        # 最终特征融合
        combined_features = torch.cat([graph_embedding, multimodal_features], dim=1)  # 24维

        # 分类
        output = self.classifier(combined_features)

        return F.log_softmax(output, dim=1), attention_weights


# 最简版本，确保一定能运行
class BasicMultimodalGNN(nn.Module):
    def __init__(self):
        super(BasicMultimodalGNN, self).__init__()

        # 基本的图卷积层
        self.conv1 = GCNConv(config.NODE_FEATURE_DIM, 16)

        # 全局注意力池化
        self.global_attention = GlobalAttention(
            nn.Sequential(
                nn.Linear(16, 8),
                nn.Tanh(),
                nn.Linear(8, 1)
            )
        )

        # 多模态特征融合
        self.mri_fc = nn.Linear(config.MRI_DIM, 4)
        self.cog_fc = nn.Linear(config.COG_DIM, 4)
        self.clin_fc = nn.Linear(config.CLIN_DIM, 4)
        self.genetic_fc = nn.Linear(config.GENETIC_DIM, 4)

        # 多模态注意力融合
        modal_dims = [4, 4, 4, 4]
        self.multimodal_attention = MultiModalAttention(modal_dims, hidden_dim=4)

        # 计算分类器输入维度
        # 图特征: 16 + 多模态特征: 4 = 20
        classifier_input_dim = 16 + 4

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

        # 图级表示 - 使用注意力池化
        graph_embedding = self.global_attention(graph_features, batch)  # 16维

        # 多模态特征处理
        mri_features = F.relu(self.mri_fc(data.mri_features))
        cog_features = F.relu(self.cog_fc(data.cog_features))
        clin_features = F.relu(self.clin_fc(data.clin_features))
        genetic_features = F.relu(self.genetic_fc(data.genetic_features))

        # 多模态注意力融合
        modalities = [mri_features, cog_features, clin_features, genetic_features]
        multimodal_features, attention_weights = self.multimodal_attention(modalities)  # 4维

        # 最终特征融合
        combined_features = torch.cat([graph_embedding, multimodal_features], dim=1)  # 20维

        # 分类
        output = self.classifier(combined_features)

        return F.log_softmax(output, dim=1), attention_weights