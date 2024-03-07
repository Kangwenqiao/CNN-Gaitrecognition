#train.py
import os

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from load_datasets import load_dataset
from network import NetWork

print('pytorch版本:', torch.__version__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 可排除一些bug

# ---------------------------------DK数据工作室：基于深度学习的步态识别---------------------------------
'''
数据集使用CASIA-B：这里下载：http://www.cbsr.ia.ac.cn/china/Gait%20Databases%20CH.asp  。找到“DataSet B”，点击下载
共124人，有3种行走状态：正常nm-01~nm-06  带背包bg-01~bg-02 穿大衣cl-01~cl-02，共10个。每个行走状态又有11个视角

我们的博客配置教程：https://blog.csdn.net/weixin_45941288/article/details/123428784?spm=1001.2014.3001.5501
之前的环境，建议卸了，以免出错，错了又要重来！然后跟着我们教程配置就行，至今没试过有失败的案例

本代码构造简单，技术已经对其优化，适合小白入门步态识别领域。如果后续有其他需要制作的，欢迎继续联系我们，我们致力于做出更高质量的代码
'''


# 画图
def draw_loss_acc(history):
    fig = plt.figure(figsize=(10, 7))
    ax1 = plt.subplot(2, 1, 1)
    plt.title('loss and val_loss')
    plt.plot(history.get('epoch'), history.get('loss'), label='loss', color='g')  # 这是训练时的损失函数
    plt.plot(history.get('epoch'), history.get('val_loss'), label='val_loss', color='r')  # 这是测试时的损失函数
    for i in range(len(history.get('epoch'))):
        if i % 20 == 0:
            plt.annotate('(%d,%.2f)' % (history.get('epoch')[i], history.get('loss')[i]),
                         xy=(history.get('epoch')[i], history.get('loss')[i]))
            plt.annotate('(%d,%.2f)' % (history.get('epoch')[i], history.get('val_loss')[i]),
                         xy=(history.get('epoch')[i], history.get('val_loss')[i]))
    plt.legend()

    ax2 = plt.subplot(2, 1, 2)
    plt.title('accuracy and val_accuracy')
    plt.plot(history.get('epoch'), history.get('accuracy'), label='accuracy', color='g')  # 这是训练时的准确率
    plt.plot(history.get('epoch'), history.get('val_accuracy'), label='val_accuracy', color='r')  # 这是测试时的准确率
    for i in range(len(history.get('epoch'))):
        if i % 20 == 0:
            plt.annotate('(%d,%.2f)' % (history.get('epoch')[i], history.get('accuracy')[i]),
                         xy=(history.get('epoch')[i], history.get('accuracy')[i]))
            plt.annotate('(%d,%.2f)' % (history.get('epoch')[i], history.get('val_accuracy')[i]),
                         xy=(history.get('epoch')[i], history.get('val_accuracy')[i]))
    plt.legend()
    plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 显卡能用于深度学习，就会使用，用不了则自动使用CPU运行

# 1、定义参数
epochs = 100  # 训练几轮，也叫迭代次数或训练次数
batch_size = 128  # 每轮每批输入的个数
n_class = 124  # 共几类
save_model_name = 'model.h5'  # 模型保存的名字
# 2、加载数据集中的训练集和测试集(需要测试集是为观察损失函数和准确率，防止过拟合等)
x_train, y_train, x_test, y_test = load_dataset(ID=(1, 124))  # 获取训练集的x和y和测试集的x和y。x是GEI图，y是标签
x_train = np.transpose(x_train, axes=[0, 3, 1, 2]).astype('float32') / 255.  # 转置和归一化
x_test = np.transpose(x_test, axes=[0, 3, 1, 2]).astype('float32') / 255.
x_train, y_train, x_test, y_test = torch.tensor(x_train), torch.tensor(y_train), torch.tensor(x_test), torch.tensor(
    y_test)
b, c, h, w = x_train.shape
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, y_train.dtype)
torch_train_dataset = torch.utils.data.TensorDataset(x_train, y_train.long())
torch_test_dataset = torch.utils.data.TensorDataset(x_test, y_test.long())
train_loader = torch.utils.data.DataLoader(dataset=torch_train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=torch_test_dataset, batch_size=batch_size,
                                          shuffle=False)

a = next(iter(train_loader))
print(a[0].shape, a[1].shape)

# 搭建网络
model = NetWork(c, h, w, n_class).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 定义优化器
# 训练
try:  # 尝试载入模型，可断点续训
    model.load_state_dict(torch.load(save_model_name, map_location=device))
    print('已加载上次的权重，如需从0开始训练，请删除.h5文件再试\n断点续训中......')
except:
    print('模型未载入，权重和网络不匹配！正在重新训练中......')
    pass
history = {'epoch': [], 'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
for epoch in range(epochs):
    # 训练
    model.train()
    train_correct = 0
    train_total = 0
    train_total_loss = 0.
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)  # (b,c,h,w)
        y = y.to(device)  # (b,)
        logits = model(x)  # (b,n_class)
        train_loss = nn.CrossEntropyLoss()(logits, y)  # 使用多分类交叉熵损失函数， y不需要onehot编码！！
        # print('loss:', loss, loss.shape)
        # train_loss = train_loss.mean()
        train_total_loss += train_loss.item()
        train_total += y.size(0)
        _, y_pred = torch.max(logits, dim=1)
        train_correct += (y_pred == y).sum().item()
        # 梯度归0反向传播
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # print('loss:', loss.item())
    # 测试
    model.eval()
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        test_total_loss = 0.
        for i_test, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            test_loss = nn.CrossEntropyLoss()(logits, y)
            test_total_loss += test_loss.item()
            _, y_pred = torch.max(logits, dim=1)
            test_total += y.size(0)
            test_correct += (y_pred == y).sum().item()

    # 记录训练损失函数loss和准确率accuracy，以及测试损失函数val_loss和准确率val_accuracy
    loss = train_total_loss / (i + 1)
    val_loss = test_total_loss / (i_test + 1)
    accuracy = train_correct / train_total
    val_accuracy = test_correct / test_total
    history['loss'].append(np.array(loss))  # 训练集损失函数
    history['val_loss'].append(np.array(val_loss))  # 测试集损失函数
    history['accuracy'].append(np.array(accuracy))  # 训练集准确率
    history['val_accuracy'].append(np.array(val_accuracy))  # 测试集准确率
    history['epoch'].append(epoch)  # 训练次数
    print('epochs:%s/%s:' % (epoch + 1, epochs),
          'loss:%.6f' % history['loss'][epoch], 'accuracy:%.6f' % history['accuracy'][epoch],
          'val_loss:%.6f' % history['val_loss'][epoch], 'val_accuracy:%.6f' % history['val_accuracy'][epoch])
# 保存模型
print('已保存模型到:%s' % save_model_name)
torch.save(model.state_dict(), save_model_name)
draw_loss_acc(history)
