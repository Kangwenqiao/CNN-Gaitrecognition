#GUI.py
# -*- coding: utf-8 -*-
import os
import threading
import tkinter as tk
import tkinter.messagebox as msgbox
from tkinter import filedialog

import numpy as np
from PIL import ImageTk, Image

from evaluate_one import evaluate
from utils import show_image


# 按键调用函数：导入路径
def import_content_path():
    img_path = filedialog.askopenfilename()
    img_path = img_path.replace('\\', '/')  # 反斜杠转为斜杠，可适用于linux系统
    img_name = img_path.split('/')[-1]  # 001_nm-01_000.png
    content_path_val.set(img_path)

    pilImage = Image.open(img_path).resize((128, 128))  # 把图片放上去
    tkImage = ImageTk.PhotoImage(image=pilImage)
    label2.configure(image=tkImage)
    label2.image = tkImage  # 除非是面向对象的写法，否则一定要加上这一句！不然界面是显示全白色图片的
    y_true = int(img_name[0:img_name.find('_')]) - 1  # 第0个label表示第1个人
    label3.configure(text='真实标签:')
    label3_num.configure(text=str(y_true))
    probLabel.configure(text='')
    numLabel.configure(text='')


# 按键调用函数：显示GEI图
def show_img():
    def A():
        img_path = content_entry.get()
        img_path = img_path.replace('\\', '/')  # 反斜杠转为斜杠，可适用于linux系统
        img_name = img_path.split('/')[-1]  # 001_nm-01_000.png
        if os.path.exists(img_path) is False:
            msgbox.showwarning('错误', '图片路径不正确，请再次输入')
        else:
            show_image(img_path, title_name=img_name, equal_scale_to=384)

    T = threading.Thread(target=A)
    T.start()


def transforms():
    def A():
        global y_pred
        img_path = content_entry.get()  # 获取待预测行人的图片的路径
        img_path = img_path.replace('\\', '/')  # 反斜杠转为斜杠，可适用于linux系统
        y_pred_prob, y_pred, y_true, acc = evaluate(img_path)  # 预测行人
        probLabel.configure(text='%.4f' % np.squeeze(y_pred_prob)[y_pred])  # 更新probLabel
        if y_pred == y_true:
            numLabel.configure(text=str(y_pred), fg="green")  # 更新numLabel
            msgbox.showinfo('完成', '步态识别已完成，预测正确！预测的行人标签为:%s' % y_pred)
        else:
            numLabel.configure(text=str(y_pred), fg="red")  # 更新numLabel
            msgbox.showwarning('完成', '步态识别已完成，预测错误！预测的行人标签为:%s，但真实标签为:%s' % (y_pred, y_true))

    T = threading.Thread(target=A)
    T.start()


# 按键调用函数：显示输出结果图
def show_result_images():  # 图片为(b,h,center,c)的uint8类型

    try:
        global y_pred
        msgbox.showinfo('完成', '识别已完成，预测为"%s"' % y_pred)
        show_img()
    except:
        msgbox.showwarning('错误', '请先识别，再显示结果')


# 定义参数（调参只需要改这里即可，其他地方不用改）
n_class = 124  # 类别数
img_path = r'./GEI/001_bg-02_000.png'  # 输入需要识别的图片的路径
h, w, c = 64, 64, 1  # 数据集图片的宽和高和通道
root_h, root_w = 380, 500  # 界面的大小
# 排除一些BUG
img_path = img_path.replace('\\', '/')  # 反斜杠转为斜杠，可适用于linux系统
img_name = img_path.split('/')[-1]  # 001_nm-01_000.png
# -----------------------------------------------GUI界面部分-----------------------------------------------
# 创建GUI总界面
root = tk.Tk()
root.geometry('%sx%s+100+200' % (root_w, root_h))  # 窗口大小
root.title('步态识别系统')  # 窗口标题
# 设置背景图
canvas = tk.Canvas(root, width=root_w, height=root_h, bd=0, highlightthickness=0)
bg_imgpath = 'background.jpg'
bg_img = Image.open(bg_imgpath).resize((root_w, root_h))
bg_photo = ImageTk.PhotoImage(bg_img)
canvas.create_image(root_w // 2, root_h // 2, image=bg_photo)
canvas.place(x=0, y=0)
# 创建容器
fm1 = tk.Frame(root)
fm2 = tk.Frame(root)
fm3 = tk.Frame(root)
fm4 = tk.Frame(root)
fm5 = tk.Frame(root)
# 标签
label1 = tk.Label(fm1, text='待预测行人的GEI图片路径: ', justify=tk.RIGHT)
# 创建打开文件夹的按键
file_img = Image.open(r'file.jpg').resize((16, 16))  # 界面显示打开文件夹的贴图
file_img = ImageTk.PhotoImage(file_img)
button1 = tk.Button(fm1, image=file_img, command=import_content_path)
# 打开文件夹的框，获取数据路径用
content_path_val = tk.StringVar(fm1, value=img_path)
content_entry = tk.Entry(fm1, textvariable=content_path_val, width=40)
# 识别结果显示在界面上
pilImage = Image.open(img_path).resize((128, 128))  # 把图片放上去
tkImage = ImageTk.PhotoImage(image=pilImage)
label2 = tk.Label(fm2, image=tkImage)
# 经过网络模型，进行转换
recognition_button = tk.Button(fm3, text='进行识别', command=transforms, cursor='hand2')
# 查看结果
y_true = int(img_name[0:img_name.find('_')]) - 1  # 第0个label表示第1个人
label3 = tk.Label(fm4, text='真实标签:', font=("黑体", 15, "bold"))
label3_num = tk.Label(fm4, text=str(y_true), font=("黑体", 15, "bold"), fg="green")
label4 = tk.Label(fm5, text='概率:', font=("黑体", 15, "bold"))
probLabel = tk.Label(fm5, text='', font=("黑体", 11, "bold"))
label5 = tk.Label(fm5, text='预测标签:', font=("黑体", 15, "bold"))
numLabel = tk.Label(fm5, text='', relief='groove', fg="red", font=("黑体", 15, "bold"))

# ----------------------------------放置所有组件----------------------------------
# 放置容器1的组件
label1.pack(side='left', anchor='center', expand=True)
content_entry.pack(side='left', anchor='center', expand=True)
button1.pack(side='left', anchor='center', expand=True)
# watch_button1.pack(side='left', anchor='center', expand=True)
fm1.pack(side='top', anchor='center', expand=True)
# 放置容器2的组件
label2.pack(side='left', anchor='center', expand=True)
fm2.pack(side='top', anchor='center', expand=True)
#  放置容器3的组件 识别按键
recognition_button.pack(side='top', anchor='center', expand=True)
fm3.pack(side='top', anchor='center', expand=True)
# 放置容器4的组件 查看结果组件
label3.pack(side='left', anchor='center')
label3_num.pack(side='left', anchor='center')
fm4.pack(side='top', anchor='center', expand=True)
# 放置容器5的组件
label4.pack(side='left', anchor='center')
probLabel.pack(side='left', anchor='center')
label5.pack(side='left', anchor='center')
numLabel.pack(side='left', anchor='center')
fm5.pack(side='top', anchor='center', expand=True)
root.mainloop()
