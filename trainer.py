import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

def train(model, train_loader, val_loader, num_epochs=50, lr=1e-3, momentum=0.9, weight_decay=1e-4, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), enable_lr_decay=True, lr_decay_step=10, lr_decay_gamma=0.1, lr_decay_threshold=0.00001):
    # 将模型移动到指定设备
    model = model.to(device)

    # 定义优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
    criterion = nn.CrossEntropyLoss()

    # 创建 TensorBoard 写入器
    writer = SummaryWriter()

    # 训练模型
    best_acc = 0.0
    for epoch in range(num_epochs):
        # 训练
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # 验证
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels.detach_()).sum().item()
        val_acc = 100 * correct / total
        val_loss = val_loss / len(val_loader)

        # 打印结果
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}')

        # 当enable_lr_decay为True且当前学习率大于lr_decay_threshold时进行学习率衰减
        if enable_lr_decay and optimizer.param_groups[0]['lr'] > lr_decay_threshold:
            scheduler.step()
            print("___lr:", optimizer.param_groups[0]['lr'])

        # 将训练损失和验证准确率写入 TensorBoard
        train_tags = ['Train-Acc', 'Train-Loss']
        val_tags = ['Val-Acc', 'Val-Loss']
        # tensorboard可视化
        for tag, value in zip(train_tags, [train_acc, train_loss]):
            writer.add_scalars(tag, {'Train': value}, epoch)

        for tag, value in zip(val_tags, [val_acc, val_loss]):
            writer.add_scalars(tag, {'Val': value}, epoch)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    # 关闭 TensorBoard 写入器
    writer.close()

    # 输出最佳模型的准确率
    print(f'Best validation accuracy: {best_acc:.2f}%')

    return model