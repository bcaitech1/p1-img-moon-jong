from torchvision.models import resnet34, vgg19_bn, resnext50_32x4d, resnet101
import torch.nn as nn
import torch.nn.functional as F
import torch


# -- F1 Loss
# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=18, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


class Resnet34(nn.Module):
    def __init__(self, num_classes=18):
        super(Resnet34, self).__init__()
        self.backbone = resnet34(pretrained=True)
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x


class Vgg19(nn.Module):
    def __init__(self, num_classes=18):
        super(Vgg19, self).__init__()
        self.backbone = vgg19_bn(pretrained=True)
        self.backbone.fc = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x


class Resnext50(nn.Module):
    def __init__(self, num_classes=18):
        super(Resnext50, self).__init__()
        self.backbone = resnext50_32x4d(pretrained=True)
        self.backbone.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x


class Resnet101(nn.Module):
    """
    Resnet101 backbone
    """

    def __init__(self, num_classes=18):
        super(Resnet101, self).__init__()
        self.net = resnet101(pretrained=True)
        self.net.fc = nn.Linear(2048, 18)

    def forward(self, x):
        x = self.net(x)
        return x


class Resnet101_8class(nn.Module):
    """
    Resnet101 backbone
    """

    def __init__(self, num_classes=8):
        super(Resnet101_8class, self).__init__()
        self.net = resnet101(pretrained=True)
        self.net.fc = nn.Linear(2048, 8)

    def forward(self, x):
        x = self.net(x)
        return x