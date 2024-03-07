#evaluate_one.py
import numpy as np
import torch

from network import NetWork
from utils import cv_imread

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 显卡能用于深度学习，就会使用，用不了则自动使用CPU运行

n_class = 124  # 共几类
save_model_name = 'model.h5'  # 模型保存的名字


def evaluate(image_path=r'./GEI/001_nm-01_000.png'):
    image_path = image_path.replace('\\', '/')  # 反斜杠转为斜杠，可适用于linux系统
    image = cv_imread(image_path)[...,0]  # 读图
    image = image[np.newaxis, :, :, np.newaxis]  # 维度扩充为(b,h,w,c)
    image = np.transpose(image, axes=[0, 3, 1, 2]).astype('float32') / 255.  # 转置和归一化
    b, c, h, w = image.shape
    image_name = image_path.split('/')[-1]  # 001_nm-01_000.png
    label = int(image_name[0:3])
    # 加载网络
    model = NetWork(c, h, w, n_class).to(device)  # 加载网络(调用network.py文件里的network函数)
    # 测试网络
    model.load_state_dict(torch.load(save_model_name, map_location=device))  # 加载刚刚保存的权重给网络
    model.eval()
    # y_pred_prob = model(image)  # 预测，和上面val_accuracy其实是一样的
    y_pred_prob = model(torch.from_numpy(image).to(device))  # 预测 (b,n_class)
    y_pred_prob = torch.softmax(y_pred_prob.float(), dim=-1).float().cpu().detach().numpy()  # (b,n_class)
    y_true = label - 1  # 从0开始编号，到123，共124人
    y_pred = np.argmax(y_pred_prob, -1).item()  # 从0开始编号，到123，共124人
    acc = np.mean(np.equal(y_pred, y_true) * 100)
    print('真实标签:', y_true)
    print('预测标签:', y_pred)
    print('预测概率:', np.squeeze(y_pred_prob)[y_pred])
    # print('准确率:%f%%' % acc)
    return y_pred_prob, y_pred, y_true, acc


if __name__ == '__main__':
    evaluate()
