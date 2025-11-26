import torch


class Config:
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据配置
    NUM_SAMPLES = 300
    TRAIN_TEST_SPLIT = 0.2
    TRAIN_VAL_SPLIT = 0.2

    # 特征维度配置
    NODE_FEATURE_DIM = 10
    MRI_DIM = 10
    COG_DIM = 8
    CLIN_DIM = 6
    GENETIC_DIM = 4

    # 模型配置
    NUM_CLASSES = 3

    # 训练配置
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 0.001


# 创建配置实例
config = Config()