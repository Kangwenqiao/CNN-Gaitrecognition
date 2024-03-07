#network.py
import numpy as np
import torch
import torch.nn as nn


# 该py文件就是整个步态识别的算法部分，也叫神经网络模型
# 整个神经网络总结为两层'卷积+激活函数+池化'。在卷积和池化之间加入标准化，使得数据值范围在0-1之间易于被模型训练，在最后使用随机丢弃层降低过拟合，提升模型准确率

class NetWork(nn.Module):  # 网络参数都是超参数，本网络由DK数据工作室调参多次，类似VGG的结构，几乎能使得所有的训练方式都能达到最高的准确率
    def __init__(self, c, h, w, n_class):
        super(NetWork, self).__init__()
        self.c, self.h, self.w, self.n_class = c, h, w, n_class
        # Conv2D的参数in_channels是输入的通道数，out_channels是卷积核的个数(也是输出通道数)，kernel_size是卷积核的大小，strides是每次卷积的步长，padding是是否进行边界填充
        self.conv1 = nn.Conv2d(in_channels=self.c, out_channels=4, kernel_size=3, stride=1, padding=1)
        # BatchNorm2d的参数num_features是输入的通道数，和上面的in_channels参数一个意思
        self.bn1 = nn.BatchNorm2d(num_features=4)  # 批归一化层，把每个批次batch的数据都进行归一化，作用是防止过拟合，增加准确率
        self.relu = nn.ReLU()  # 激活函数使用最常用的relu函数
        self.maxpooling2d = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层，使得特征图的宽和高都会除以2，作用是提取出更重要的特征，不重要的特征直接丢弃
        self.dropout2d = nn.Dropout2d(0.3)  # 改进的Dropout层，作用是更好地防止过拟合，提升准确率

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=8)

        self.flatten = nn.Flatten()  # 把宽高和通道打平成一起  -->  (b,c*h*w)
        self.linear = nn.Linear(in_features=int(np.prod([self.h // 4, self.w // 4, 8])), out_features=self.n_class)

    def forward(self, inputs):  # 整个神经网络总结为两层'卷积+激活函数+池化'。在卷积和池化之间加入标准化层，使得数据值范围在0-1之间易于被模型训练，在最后使用随机丢弃层降低过拟合，提升模型准确率
        # 第一层
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpooling2d(x)
        x = self.dropout2d(x)
        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpooling2d(x)
        x = self.dropout2d(x)
        # 对特征进行打平成向量，然后使用全连接层进行分类输出
        x = self.flatten(x)  # 打平层
        x = self.linear(x)  # 全连接层，也是输出层
        return x


def main():  # 这里是起到调试神经网络的作用
    b = 128  # batch_size，是指一次输入多少张GEI能量图到网络里
    c, h, w, n_class = 1, 64, 64, 124
    x = torch.ones([b, c, h, w])
    model = NetWork(c, h, w, n_class)
    print(model(x).shape)


if __name__ == '__main__':
    main()
