import torch
import torch.nn as nn
import torchvision.models as models

#注意力机制模块
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = in_channels // reduction
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        scale = self.fc(x).view(b, c, 1, 1)
        return x * scale

#辅助分类器模块
class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels=1056, num_classes=100):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.head(x)

#DenseNet-SEDA 主模型
class DenseNet_SEDA(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()
        weights = models.DenseNet161_Weights.DEFAULT if pretrained else None
        densenet = models.densenet161(weights=weights)
        f = densenet.features

        self.stage1 = nn.Sequential(
            f.conv0, f.norm0, f.relu0, f.pool0,
            f.denseblock1, f.transition1,
            f.denseblock2, f.transition2,
            f.denseblock3, f.transition3
        )
        self.aux_classifier = AuxiliaryClassifier(in_channels=1056, num_classes=num_classes)

        self.stage2 = nn.Sequential(
            f.denseblock4,
            f.norm5
        )
        self.se = SqueezeExcitation(in_channels=2208, reduction=16)

        flatten_dim = 2208 * 7 * 7
        self.primary_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        mid = self.stage1(x)
        feat = self.stage2(mid)
        feat = self.se(feat)
        out_main = self.primary_classifier(feat)

        if self.training:
            out_aux = self.aux_classifier(mid)
            return out_main, out_aux
        return out_main

if __name__ == "__main__":
    net = DenseNet_SEDA(num_classes=100, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    net.train()
    m, a = net(x)
    print(f"Train outputs: Main {m.shape}, Aux {a.shape}")
    net.eval()
    with torch.no_grad():
        out = net(x)
    print(f"Eval output: {out.shape}")