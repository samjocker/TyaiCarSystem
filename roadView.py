import cv2
import numpy as np
import tensorflow as tf
from PIL import Image,ImageOps
from PIL import ImageFilter

import colorsys
import copy
import time, sys, json
import threading

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *

from nets.deeplab import Deeplabv3
from utils.utils import cvtColor, preprocess_input, resize_image

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
MainWindow.setObjectName("MainWindow")
MainWindow.setWindowTitle("TYAI car")
MainWindow.resize(700, 500)

label = QtWidgets.QLabel(MainWindow)
label.setGeometry(0, 0, 720, 405)

result_label = QtWidgets.QLabel(MainWindow)
result_label.setGeometry(520, 405, 250, 160)
result_label.setStyleSheet("QLabel { background-color : white; color : black; }")
result_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
          (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
          (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
          (128, 64, 12)]

class DeeplabV3(object):
    _defaults = {
        "model_path": 'model\\ep100-loss0.153-val_loss0.047.h5',
        "num_classes": 7,
        "backbone": "mobilenet",
        "input_shape": [512, 512],
        "downsample_factor": 16,
        "mix_type": 0,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.num_classes <= 21:
            self.colors = colors
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

    def generate(self):
        self.model = Deeplabv3([self.input_shape[0], self.input_shape[1], 3], self.num_classes,
                               backbone=self.backbone, downsample_factor=self.downsample_factor)
        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))

    @tf.function
    def get_pred(self, image_data):
        pr = self.model(image_data, training=False)
        return pr

    def detect_image(self, image):
        image = cvtColor(image)

        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        image_data = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        pr = self.get_pred(image_data)[0].numpy()

        pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
            int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

        pr = pr.argmax(axis=-1)

        # Create a mask for class 1
        mask_class1 = (pr == 1).astype(np.uint8)

        # Apply the mask to the original image
        seg_img = (np.expand_dims(mask_class1, -1) * np.array(old_img, np.float32)).astype('uint8')

        image = Image.fromarray(np.uint8(seg_img))

        image = Image.blend(old_img, image, 0.7)

        # Create a new image showing only class 1
        seg_img_class1 = (np.expand_dims(pr == 1, -1) * np.array(old_img, np.float32)).astype('uint8')
        image2 = Image.fromarray(np.uint8(seg_img_class1))

        return image, pr, image2


deeplab = DeeplabV3()

video_path = "D:\\Data\\project\\tyaiCar\\TyaiCarSystem\\test5.mp4"
video_save_path = ""
video_fps = 30.0

def perspective_correction(image):
    # 定義原始四邊形的四個點
    original_points = np.float32([[-400, 480], [1120, 480], [200, 280], [520, 280]])

    # 定義梯形校正後的四個點
    corrected_points = np.float32([[200, 480], [520, 480], [200, 0], [520, 0]])

    # 計算透視變換矩陣
    perspective_matrix = cv2.getPerspectiveTransform(original_points, corrected_points)

    # 執行透視變換
    result = cv2.warpPerspective(image, perspective_matrix, (720, 480))

    return result


trapezoid_label = QtWidgets.QLabel(MainWindow)
trapezoid_label.setGeometry(440, 0, 250, 160)
trapezoid_label.setStyleSheet("QLabel { background-color : white; color : black; }")
trapezoid_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)


def opencv():
    capture = cv2.VideoCapture(video_path)
    if video_save_path != "":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    ref, frame = capture.read()
    if not ref:
        raise ValueError("Video source Error")

    fps = 0.0
    while True:
        t1 = time.time()
        ref, frame = capture.read()
        ref, frame = capture.read()
        ref, frame = capture.read()


        if not ref:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))

        deeplab.mix_type = 0
        result_img_blend, rd,result_img_trapezoid2 = deeplab.detect_image(frame)
        # print(type(result_img_trapezoid2))
        # deeplab.mix_type = 1
        # result_img_no_blend, _ = deeplab.detect_image(frame)

        height, width, channel = 405, 720, 3

        # 上方混合模式結果
        frame_blend = cv2.resize(np.array(result_img_blend), (width, height))
        bytesPerline_blend = channel * width
        img_blend = QImage(frame_blend, width, height, bytesPerline_blend, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img_blend))


        # 开始绘制
        painter = QPainter(label.pixmap())
        font = QFont()
        font.setPointSize(15)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))  # 文字顏色，白色
        painter.drawText(20, 30, f"FPS: {fps}")  # 在左上角顯示FPS小數點後兩位
        #painter.drawText(20, 50, f"Person:")

        # 结束绘制
        painter.end()

        # gray_image = ImageOps.grayscale(result_img_trapezoid2)

        # # 使用邊緣檢測算法，例如Canny
        # edge_image = gray_image.filter(ImageFilter.FIND_EDGES)


        # # 下方不混合模式結果
        # result_img_no_blend_np = np.array(edge_image.resize((200, 160), Image.BICUBIC))
        # bytesPerline_no_blend = 3 * result_img_no_blend_np.shape[1]
        # result_img_no_blend_qt = QImage(result_img_no_blend_np.data, result_img_no_blend_np.shape[1],
        #                                 result_img_no_blend_np.shape[0], bytesPerline_no_blend, QImage.Format_RGB888)
        # result_label.setPixmap(QPixmap.fromImage(result_img_no_blend_qt))


        # 梯形校正
        result_img_trapezoid_corrected = perspective_correction(np.array(result_img_trapezoid2))

        # 下方梯形校正結果
       #print(type(result_img_trapezoid_corrected))


        result_img_trapezoid_corrected_pil = Image.fromarray(result_img_trapezoid_corrected)
        result_img_trapezoid_np = np.array(result_img_trapezoid_corrected_pil.resize((250, 160), Image.BICUBIC))

        bytesPerline_trapezoid = 3 * result_img_trapezoid_np.shape[1]
        result_img_trapezoid_qt = QImage(result_img_trapezoid_np.data, result_img_trapezoid_np.shape[1],
                                        result_img_trapezoid_np.shape[0], bytesPerline_trapezoid, QImage.Format_RGB888)
        trapezoid_label.setPixmap(QPixmap.fromImage(result_img_trapezoid_qt))


        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %.2f"%(fps), end='\r')

        if video_save_path != "":
            out.write(frame)

video = threading.Thread(target=opencv)
video.start()

MainWindow.show()
sys.exit(app.exec_())