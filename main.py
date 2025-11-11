import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from models import BasicMultimodalGNN  # 使用最简版本确保能运行
from data_utils import create_improved_synthetic_dataset, prepare_data_loaders, get_dataset_info
from train import improved_train_model, detailed_evaluate_model
from config import config


def main():
    print("=== 基于多模态GNN的阿尔茨海默病风险预测 ===")
    print(f"使用设备: {config.DEVICE}")

    # 1. 创建数据集
    print("\n1. 创建数据集...")
    dataset = create_improved_synthetic_dataset()
    train_loader, val_loader, test_loader, train_data, val_data, test_data = prepare_data_loaders(dataset)

    # 显示数据集信息
    info = get_dataset_info(dataset, train_data, val_data, test_data)
    print(f"总样本数: {info['total_samples']}")
    print(f"训练集: {info['train_samples']} 样本")
    print(f"验证集: {info['val_samples']} 样本")
    print(f"测试集: {info['test_samples']} 样本")
    print(f"类别分布: {info['class_distribution']}")

    # 2. 初始化模型
    print("\n2. 初始化模型...")
    model = BasicMultimodalGNN().to(config.DEVICE)  # 使用最简模型
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 打印模型结构以便调试
    print("\n模型结构:")
    print(model)

    # 3. 训练模型
    print("\n3. 开始训练...")
    train_losses, val_accuracies = improved_train_model(model, train_loader, val_loader)

    # 4. 评估模型
    print("\n4. 模型评估...")
    test_accuracy, test_auc, cm, preds, labels, class_report = detailed_evaluate_model(model, test_loader)

    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集AUC: {test_auc:.4f}")
    print("\n详细分类报告:")
    print(class_report)
    print("\n混淆矩阵:")
    print(cm)

    # 5. 可视化结果
    print("\n5. 生成可视化结果...")
    plot_results(train_losses, val_accuracies, cm)

    # 6. 风险预测示例
    print("\n6. 风险预测示例...")
    show_prediction_examples(model, test_data)


def plot_results(train_losses, val_accuracies, cm):
    """绘制结果图表"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 3, 3)
    classes = ['Normal', 'MCI', 'AD']
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def show_prediction_examples(model, test_data):
    """显示预测示例"""
    class_names = ['正常', '轻度认知障碍(MCI)', '阿尔茨海默病(AD)']

    # 使用正确的DataLoader
    from torch_geometric.loader import DataLoader
    single_loader = DataLoader(test_data[:3], batch_size=1, shuffle=False)

    print("风险预测示例:")
    for i, batch in enumerate(single_loader):
        batch = batch.to(config.DEVICE)
        model.eval()
        with torch.no_grad():
            output = model(batch)
            probabilities = torch.exp(output).cpu().numpy()[0]

            predicted_class = output.argmax(dim=1).item()
            true_class = batch.y[0].item()

            print(f"\n样本 {i + 1}:")
            print("预测概率分布:")
            for j, class_name in enumerate(class_names):
                print(f"  {class_name}: {probabilities[j]:.4f}")

            print(f"真实标签: {class_names[true_class]}")
            print(f"预测标签: {class_names[predicted_class]}")
            print(f"{'✓ 正确' if predicted_class == true_class else '✗ 错误'}")


if __name__ == "__main__":
    main()