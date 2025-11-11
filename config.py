# 配置文件
import torch


class Config:
    # 数据配置
    NUM_SAMPLES = 200  # 回到原来的样本量
    NODE_FEATURE_DIM = 16  # 回到原来的维度
    NUM_CLASSES = 3

    # 模型配置 - 使用更保守的设置
    HIDDEN_DIM = 32
    DROPOUT = 0.3

    # 训练配置
    BATCH_SIZE = 8  # 回到原来的批次大小
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EPOCHS = 100
    TRAIN_TEST_SPLIT = 0.3
    TRAIN_VAL_SPLIT = 0.2

    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 特征维度
    MRI_DIM = 10
    COG_DIM = 8
    CLIN_DIM = 6
    GENETIC_DIM = 4


config = Config()