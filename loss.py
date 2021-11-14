import paddle
import paddle.nn as nn
import numpy as np


class SimiLoss(nn.Layer):
    def __init__(self, cls=10, margin=100):
        super(SimiLoss, self).__init__()
        self.cls = cls
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x = paddle.flatten(x,1,-1)  # [N,C]
        simi_metric = paddle.sum((x.unsqueeze(0) - x.unsqueeze(1)) ** 2, axis=-1)  # [N,N]
        J = 0
        m = x.shape[0]
        for i in range(m):
            pos = paddle.cast(y == y[i], 'float32')  # [N]
            neg = paddle.cast(y != y[i], 'float32')  # [N]
            J += paddle.sum(pos * simi_metric[i, :]) + paddle.sum(neg * self.relu(self.margin - simi_metric[i, :]))
        return J / (2 * m**2)
        
if __name__ == '__main__':
    x = paddle.randn([4,3,224,224])
    x.stop_gradient = False
    label = paddle.to_tensor([0,3,5,0])
    loss = SimiLoss(margin=100)
    print(loss(x,label))
    # print(x.gradient)



