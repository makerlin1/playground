import paddle
import paddle.nn as nn
import numpy as np
from paddle.vision import transforms
from blocks import *
from loss import *
from paddle.vision.datasets import Cifar10
from paddle.vision.transforms import Normalize, Transpose, Compose
from paddle.io import DataLoader


transform = Compose([Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    data_format='HWC'),
                    Transpose()])
cifar10 = Cifar10(mode='train', transform=transform)  # dataset
cifar10_te = Cifar10(mode='test', transform=transform)  # dataset

channel_list = [32, 64, 128, 256, 512]
SMNet = StackMobileNet(num_classes=10)
SMNet.add_stack(channel_list.pop(0))
loss = SimiLoss(cls=10,margin=1000)
net = paddle.Model(SMNet)
# clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
net.prepare(optimizer=paddle.optimizer.Adam(parameters=net.parameters()),loss=SimiLoss(cls=10,margin=1000))
net.fit(train_data=cifar10,
        epochs=12,
        batch_size=8,
        verbose=1)




    
                    