import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from config import config


def validate_model(model, val_loader):
    """验证模型"""
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(config.DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            val_preds.extend(pred.cpu().numpy())
            val_labels.extend(data.y.squeeze().cpu().numpy())

    return accuracy_score(val_labels, val_preds)


def improved_train_model(model, train_loader, val_loader):
    """改进的训练模型"""
    # 使用AdamW优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=1e-5
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    patience = 20
    counter = 0

    print("开始训练...")
    for epoch in range(config.EPOCHS):
        # 训练阶段
        model.train()
        total_loss = 0
        batch_count = 0

        for data in train_loader:
            data = data.to(config.DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y.squeeze())
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        scheduler.step()

        avg_loss = total_loss / batch_count
        train_losses.append(avg_loss)

        # 验证阶段
        val_acc = validate_model(model, val_loader)
        val_accuracies.append(val_acc)

        # 早停法
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1

        if counter >= patience:
            print(f"早停于第 {epoch + 1} 轮")
            break

        if (epoch + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'轮次 {epoch + 1:03d}, 损失: {avg_loss:.4f}, 验证准确率: {val_acc:.4f}, 学习率: {current_lr:.6f}')

    # 加载最佳模型
    try:
        model.load_state_dict(torch.load('best_model.pth'))
        print("已加载最佳模型")
    except:
        print("警告: 无法加载最佳模型，使用当前模型")

    return train_losses, val_accuracies


def detailed_evaluate_model(model, test_loader):
    """详细的模型评估"""
    model.eval()
    test_preds = []
    test_probs = []
    test_labels = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(config.DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            test_preds.extend(pred.cpu().numpy())
            test_probs.extend(torch.exp(output).cpu().numpy())
            test_labels.extend(data.y.squeeze().cpu().numpy())

    accuracy = accuracy_score(test_labels, test_preds)

    # 多分类AUC
    try:
        auc = roc_auc_score(test_labels, test_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0

    cm = confusion_matrix(test_labels, test_preds)

    # 详细分类报告
    class_report = classification_report(test_labels, test_preds,
                                         target_names=['Normal', 'MCI', 'AD'])

    return accuracy, auc, cm, test_preds, test_labels, class_report


# 兼容性函数
def train_model(model, train_loader, val_loader):
    """原始训练函数"""
    return improved_train_model(model, train_loader, val_loader)


def evaluate_model(model, test_loader):
    """原始评估函数"""
    accuracy, auc, cm, preds, labels, _ = detailed_evaluate_model(model, test_loader)
    return accuracy, auc, cm, preds, labels