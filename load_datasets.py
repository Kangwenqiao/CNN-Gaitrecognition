#load_datasets.py
import os

import cv2 as cv
import numpy as np

'''
数据集使用CASIA-B：共124人，有3种行走状态：正常nm-01~nm-06  带背包bg-01~bg-02 穿大衣cl-01~cl-02，共10个。每个行走状态又有11个视角
'''


def load_dataset(dir_path=r'.\GEI', ID=(0, 124)):  # 这里是加载数据集，默认124人全部加载
    x_train_list, x_test_list, y_train_list, y_test_list = [], [], [], []
    image_names = os.listdir(dir_path)
    for image_name in image_names:
        image = cv.imread(os.path.join(dir_path, image_name), flags=0)#以灰图的形式读入
        image=image[:,:,np.newaxis]
        label = int(image_name[0:image_name.find('_')]) - 1  # 第0 label表示第1个人
        index = image_name[0:3]
        condition = image_name[4:9]  # 找出nm-01这些
        angle = image_name[10:13]

        # 以下是训练集和预测集，是通过行走条件划分的, 可根据自己需要修改(保证训练集和测试集没有重复就都是对的)
        if ID[0] <= int(index) <= ID[1]:  # 注释的是常用的训练方式，解注释即可使用(小白提示：#号就是注释，删掉#号就是解注释，解注释后原本的那行需要加回注释)
            # 训练集
            #if condition == 'nm-01' or condition == 'nm-02' or condition == 'nm-03' or condition == 'nm-04':  # 仿照这种写法即可自己改
            if condition == 'nm-01' or condition == 'nm-02' or condition == 'nm-03' or condition == 'nm-04' or condition == 'bg-01' or condition == 'cl-01':
            #if condition == 'nm-01' or condition == 'nm-02' or condition == 'nm-03' or condition == 'nm-04':
            #if condition == 'nm-01' or condition == 'nm-02' or condition == 'nm-03' or condition == 'nm-04':
            #if condition == 'bg-01':
            #if condition == 'cl-01':
                x_train_list.append(image)
                y_train_list.append(label)
            # 测试集
            #if condition == 'nm-05' or condition == 'nm-06':
            if condition == 'nm-05' or condition == 'nm-06' or condition == 'bg-02' or condition == 'cl-02':
            #if condition == 'bg-01' or condition == 'bg-02':
            #if condition == 'cl-01' or condition == 'cl-02':
            #if condition == 'bg-02':
            #if condition == 'cl-02':
                x_test_list.append(image)
                y_test_list.append(label)
    return [np.array(x_train_list), np.array(y_train_list), np.array(x_test_list), np.array(y_test_list)]


def main():
    load_dataset()


if __name__ == '__main__':
    main()