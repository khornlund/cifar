from collections import OrderedDict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from cifar.base import BaseModel


class ResNet18(BaseModel):

    def __init__(self, num_classes, verbose=0):
        super().__init__(verbose=verbose)
        self.num_classes = num_classes

        self.encoder = models.resnet18(pretrained=True)
        num_feats = self.encoder.fc.out_features
        self.classifier = nn.Sequential(nn.Linear(num_feats, num_classes))

        # Init of last layer
        for m in self.classifier:
            nn.init.kaiming_normal_(m.weight)
        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DenseNet121(BaseModel):

    def __init__(self, num_classes, verbose=0):
        super().__init__(verbose=verbose)
        self.num_classes = num_classes

        self.encoder = models.densenet121(pretrained=True)
        num_feats = self.encoder.classifier.in_features
        classifier = nn.Sequential(nn.Linear(num_feats, num_classes))

        # Init of last layer
        for m in classifier:
            nn.init.kaiming_normal_(m.weight)
        self.logger.info(f'<init>: \n{self}')

        self.encoder.classifier = classifier

    def forward(self, x):
        x = self.encoder(x)
        return x


class DenseNet169(BaseModel):

    def __init__(self, num_classes, verbose=0):
        super().__init__(verbose=verbose)
        self.num_classes = num_classes

        self.encoder = models.densenet169(pretrained=True)
        num_feats = self.encoder.classifier.in_features
        classifier = nn.Sequential(nn.Linear(num_feats, num_classes))

        # Init of last layer
        for m in classifier:
            nn.init.kaiming_normal_(m.weight)
        self.logger.info(f'<init>: \n{self}')

        self.encoder.classifier = classifier

    def forward(self, x):
        x = self.encoder(x)
        return x


class ResNet18MaxAvg(BaseModel):

    def __init__(self, num_classes, dropout=0.5, verbose=0):
        super(ResNet18MaxAvg, self).__init__(verbose=verbose)
        self.num_classes = num_classes
        encoder = models.resnet18(pretrained=True)

        self.layer0 = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool)
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.ada_avgpool = nn.AdaptiveAvgPool2d(1)
        self.ada_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(1024)),
            ('drop1', nn.Dropout(p=dropout)),
            ('linear1', nn.Linear(1024, 512)),
            ('relu1', nn.ReLU()),
            ('bn2', nn.BatchNorm1d(512)),
            ('drop2', nn.Dropout(p=dropout)),
            ('linear2', nn.Linear(512, num_classes))
        ]))

        nn.init.kaiming_normal_(self.fc._modules['linear1'].weight)
        nn.init.kaiming_normal_(self.fc._modules['linear2'].weight)

        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        avg_x = self.ada_avgpool(x)
        max_x = self.ada_maxpool(x)
        x = torch.cat((avg_x, max_x), dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# -- Wide ResNet --
# https://github.com/uoguelph-mlrg/Cutout/blob/master/model/wide_resnet.py

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and
                                stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(BaseModel):

    def __init__(self, depth, num_classes, widen_factor=1, dropout=0.0, verbose=2):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))

        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out
