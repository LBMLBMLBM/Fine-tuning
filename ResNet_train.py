from loaddata import cub200
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import ResNet18
from trainer import train

# 定义数据预处理
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载训练和验证数据集
train_dataset = cub200(root='../data', train=True, transform=transform_train)
val_dataset = cub200(root='../data', train=False, transform=transform_val)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

pre_trained_model = ResNet18(pre_trained=True, n_class=200)
train(pre_trained_model, train_loader, val_loader, num_epochs=50, lr=0.01, enable_lr_decay = True, lr_decay_step=20, lr_decay_threshold=0.0001)

# 所有参数设计为可训练
for param in pre_trained_model.parameters():
        param.requires_grad = True;

# 使用较小的学习率，对其余参数进行训练
train(pre_trained_model, train_loader, val_loader, num_epochs=50, lr=0.01, weight_decay=1e-2, enable_lr_decay = True, lr_decay_step=20, lr_decay_threshold=0.0001)
