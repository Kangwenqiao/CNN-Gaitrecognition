#test.py
import time

import numpy as np
import os
import torch

from load_datasets import load_dataset
from network import NetWork

'''
这里是测试整个测试集的准确率。输入每个人的GEI图，输出各自的每个类别的概率，然后选出概率最大的类别作为识别的类别
数据集使用CASIA-B：共124人，有3种行走状态：正常nm-01~nm-06  带背包bg-01~bg-02 穿大衣cl-01~cl-02，共10个。每个行走状态又有11个视角
'''
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 可排除一些bug
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 显卡能用于深度学习，就会使用，用不了则自动使用CPU运行

# 定义参数
n_class = 124  # 共几类
save_model_name = 'model.h5'  # 模型保存的名字
# 加载测试集
_, _, x_test, y_test = load_dataset(
    ID=(1, 124))  # 这里默认是预测整个数据集124个人。ID是从1开始编号的，如果需要测试第1个人，就改成ID=(1, 1)。测试第2个人，就改成ID=(2, 2)
b, h, w, c = x_test.shape
x_test = np.transpose(x_test, axes=[0, 3, 1, 2]).astype('float32') / 255.  # 转置和归一化
print(x_test.shape, y_test.shape)
model = NetWork(c, h, w, n_class).to(device)  # 加载网络(调用network.py文件里的network函数)
try:  # 尝试载入模型
    model.load_state_dict(torch.load(save_model_name, map_location=device))
    print('载入训练好的模型的权重')
except:
    print('模型权重未载入，权重和网络不匹配！请重新训练！')
model.eval()
t = time.time()
y_pred = model(torch.from_numpy(x_test).to(device)).cpu().detach().numpy()  # 预测
y_pred = np.argmax(y_pred, -1)
print('预测值:', y_pred.tolist())  # 列表里的每个数值，就是行人的编号，范围从0开始数，到123，共124人。预测和真实值的编号对上，说明预测对了，否则就是预测错了
print('真实值:', y_test.tolist())
print('test_acc:', np.mean(np.equal(y_pred, y_test)), '耗时:%s秒' % (time.time() - t))  # 这个准确率就是结果，表示预测对的人，占总人数的比例
