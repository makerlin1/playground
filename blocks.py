import paddle
import paddle.nn as nn
class ConvReLUBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 kernel_size,
                 stride=1,
                 padding=1,
                 groups=1):
        super().__init__()
        self.conv = nn.Conv2D(num_channels, num_filters, kernel_size, stride, groups=groups, padding=padding)
        self.bn = nn.BatchNorm(num_filters)
        self.relu = nn.ReLU6()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return self.bn(x)

class InvertedResidualBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 kernal_size,
                 stride,
                 expand_ratio=6):
        super().__init__()
        self.middle_channels = int(num_channels * expand_ratio)
        self.conv1x1 = nn.Conv2D(num_channels, self.middle_channels, 1, stride=1)
        self.depthwise = ConvReLUBNLayer(self.middle_channels, 
                                        self.middle_channels, 
                                        kernal_size, 
                                        stride=stride,
                                        groups=self.middle_channels)
        self.conv1x1_2 = nn.Conv2D(self.middle_channels, num_filters, 1, stride=1)
        if stride != 1:
            self.shortcut = False
        else:
            self.shortcut = True
    
    def forward(self, x):
        capacity = x
        x = self.conv1x1(x)
        x = self.depthwise(x)
        x = self.conv1x1_2(x)
        if self.shortcut:
            x = x + capacity
        return x


class Stack(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 expand_ratio=6):
        super().__init__()
        self.stack1 = InvertedResidualBlock(num_channels, num_filters, 7, 2, expand_ratio)
        self.stack2 = InvertedResidualBlock(num_filters, num_filters, 3, 1, expand_ratio)

    def forward(self, x):
        x = self.stack1(x)
        x = self.stack2(x)
        return x




class StackMobileNet(nn.Layer):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stack_list = []
        self.output_channels = 3

    def add_stack(self, target_channels):
        self.stack_list.append(Stack(self.output_channels, target_channels))
        self.output_channels = target_channels
        self.stack_list = nn.LayerList(self.stack_list)

    def forward(self, x):
        for stack in self.stack_list:
            x = stack(x)
        return x

if __name__ == '__main__':
    SMNet = StackMobileNet()
    SMNet.add_stack(32)
    SMNet.add_stack(128)
    SMNet.add_stack(256)
    net = paddle.Model(SMNet)
    info = net.summary(input_size=(2, 3, 224, 224))
    print(info)


        
        



