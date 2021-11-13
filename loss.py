import paddle
import paddle.nn as nn
import numpy as np

class ContrastLoss(nn.Layer):
    def __init__(self, cls = 1000, lmb = 5):
        super(ContrastLoss, self).__init__()
        self.lmb = lmb
        self.cls = cls
    def forward(self, x, label):
        x = paddle.flatten(x,1,-1)
        center_list = []
        internal_loss = 0
        cls = 0
        for i in range(self.cls):
            ind = (label.numpy()== i).astype(np.int64)
            ind = paddle.to_tensor(ind)
            if ind.sum() == 0:
                continue
            cls += 1
            v = paddle.index_select(x, ind)  # 筛选出对应类样本
            c = paddle.mean(v,axis=0).reshape([1,-1])  # 类中心
            center_list.append(c)
            internal_loss += paddle.sum((v - c) ** 2) ** 0.5
        center_mat = paddle.stack(center_list).squeeze()
        center_mat = paddle.sum((center_mat.unsqueeze(0) - center_mat.unsqueeze(1))**2,axis=-1) ** 0.5
        center_dis = paddle.triu(center_mat).mean()
        J = center_dis - self.lmb * internal_loss/cls
        return J





if __name__ == '__main__':
    x = paddle.randn([1,3,224,224])
    x.stop_gradient = False
    label = paddle.to_tensor([0])
    loss = ContrastLoss()
    loss(x,label)
    # print(x.gradient)