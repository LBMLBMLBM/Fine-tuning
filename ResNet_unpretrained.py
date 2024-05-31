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

model = ResNet18(pre_trained=False, n_class=200)

# 所有参数设计为可训练
for param in model.parameters():
        param.requires_grad = True;

# 训练模型
train(model, train_loader, val_loader, num_epochs=50, lr=0.01, enable_lr_decay = True, lr_decay_step=20, lr_decay_threshold=0.0001)
