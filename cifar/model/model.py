from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
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

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(BaseModel):
    def __init__(self, num_classes, depth=28, widen_factor=10, dropout_rate=0.3, verbose=0):
        super(Wide_ResNet, self).__init__(verbose=verbose)
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

        self.logger.info(f'<init>: \n{self}')

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
