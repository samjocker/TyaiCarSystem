from flask import Flask, request, jsonify

app_ai = Flask(__name__)

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

# 初始化 DeeplabV3 物件
deeplab = DeeplabV3()

# 定義處理影像的路由
@app_ai.route('/process_image', methods=['POST'])
def process_image():
    try:
        # 從 POST 請求中獲取影像資料
        image_data = request.form['image']

        # 解碼 base64 影像資料
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 在這裡使用 DeepLab 模型進行影像分割
        result_img, _, _ = deeplab.detect_image(image)

        # 將結果轉換為 JSON 格式並返回給顯示端
        return jsonify({'result_image': result_img.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # 這裡需要替換成你想要的 IP 和端口
    app_ai.run(host='0.0.0.0', port=5001)
