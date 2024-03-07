#utils.py
import cv2 as cv
import numpy as np
# from torchvision.transforms import Resize

global k2
k2 = None


def cv_imread(img_path):  # 该函数用于支持输入中文的路径
    '''
    输入图片路径，输出(h,w,3)的BGR图片。BGRA图片会自动转为BGR（只能读图片，可支持中文路径）
    '''
    img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)  # 可支持中文路径
    if img.shape[-1] >= 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)  # 32位深BGRA图片转为24位BGR图
    return img


def show_image(image_or_image_path, title_name='image', equal_scale_to=None):  # 该函数用于显示图片
    '''

    :param image_or_image_path: 图片或路径，图片是(h,w,c)的uint8
    :param title_name: 图片窗口名字
    :param equal_scale_to: 等比缩放为
    :return: 无
    '''
    if isinstance(image_or_image_path, str):
        image = cv_imread(image_or_image_path)
    else:
        image = image_or_image_path
    image = resize_image(image[np.newaxis, ...], equal_scale_to=equal_scale_to)
    image = np.array(image)[0]
    # print('image', image.shape, image.dtype, image.max(), image.min())
    image = image.astype('uint8')
    cv.imshow(title_name, image)
    cv.waitKey()
    cv.destroyAllWindows()


def resize_image(images, equal_scale_to=512):  # 该函数用于缩放图片到固定尺寸
    '''
    images为(b,h,w,c)，输出改变形状后的images
    equal_scale_to为None时不改变size
    '''
    if equal_scale_to is not None:
        # images = Resize(size=(equal_scale_to, equal_scale_to))(images)
        image_list = []
        for image in images:
            image = cv.resize(image, dsize=(equal_scale_to, equal_scale_to))
            image_list.append(image)
        images = np.array(image_list)
    return images


# 从视频里截图
def screenshot_from_camera(save_name='image'):  # 按s截图，或者点击鼠标左键截图
    '''
    读视频
    :param normalize=False输出0-255的uint8；normalize=True输出0-1的float32
    :return: images, fps
    '''
    # cv.namedWindow('screenshot_from_camera')
    global k2
    capture = cv.VideoCapture(0, cv.CAP_DSHOW)  # 参数为0表示打开摄像头
    i = 0
    while 1:
        ret, frame = capture.read()
        if ret is False:
            break
        frame = cv.flip(frame, 1)
        cv.imshow('frame', frame)
        k1 = cv.waitKey(1)  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
        cv.setMouseCallback('frame', on_mouse)
        if k1 == ord('s') or k2 == ord('s'):  # 若检测到按键 ‘s’，或鼠标左键点击，则截图
            i += 1
            save_path = r'%s_%s.jpg' % (save_name, i)
            cv.imwrite(save_path, frame)
            print(r'截图已完成，已保存为:%s' % save_path)
            k2 = None
        if k1 == ord('q') or k2 == ord('q'):
            k2 = None
            break
    capture.release()
    cv.destroyAllWindows()


def on_mouse(event, x, y, flags, param):  # 鼠标点击事件
    global k2
    if event == cv.EVENT_LBUTTONDOWN:  # 鼠标左键，运行此函数
        k2 = ord('s')
    if event == cv.EVENT_RBUTTONDOWN:  # 鼠标右键，运行此函数
        k2 = ord('q')
