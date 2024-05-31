import torch.nn as nn
from utils.weight_init import weight_init_kaiming
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, pre_trained=True, n_class=200):
        super(ResNet18, self).__init__()
        self.n_class = n_class
        self.base_model = models.resnet18(pretrained=pre_trained)
        self.base_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base_model.fc = nn.Linear(512 , n_class)
        self.base_model.fc.apply(weight_init_kaiming)

        # 仅设置最后一层的参数可以训练
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.fc.weight.requires_grad = True
        self.base_model.fc.bias.requires_grad = True

    def forward(self, x):
        N = x.size(0)
        assert x.size() == (N, 3, 224, 224)
        x = self.base_model(x)
        assert x.size() == (N, self.n_class)
        return x


