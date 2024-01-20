import time

import cv2
import numpy as np
import os
import tensorflow as tf
from PIL import Image

import colorsys
import copy
import time, sys
import threading

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *

from nets.deeplab import Deeplabv3
from utils.utils import cvtColor, preprocess_input, resize_image

import json

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
MainWindow.setObjectName("MainWindow")
MainWindow.setWindowTitle("oxxo.studio")
MainWindow.resize(720, 505)

label = QtWidgets.QLabel(MainWindow)    # 建立 QLabel
label.setGeometry(0,0,720,405)          # 設定 QLabel 位置尺寸
datumXslider = QtWidgets.QSlider(MainWindow)
datumXslider.setGeometry(110, 430, 500, 30)
datumXslider.setOrientation(QtCore.Qt.Horizontal)
datumXslider.setMaximum(1920)
datumXtitle = QtWidgets.QLabel(MainWindow)
datumXtitle.setGeometry(70, 427, 80, 30)
font = QFont() 
font.setPointSize(18)
datumXtitle.setText("X 軸")
datumXtitle.setFont(font)
datumXvalue = QtWidgets.QLabel(MainWindow)
datumXvalue.setGeometry(620, 427, 80, 30)
font = QFont() 
font.setPointSize(18)
datumXvalue.setText("555")
datumXvalue.setFont(font)

datumYslider = QtWidgets.QSlider(MainWindow)
datumYslider.setGeometry(110, 460, 500, 30)
datumYslider.setOrientation(QtCore.Qt.Horizontal)
datumYslider.setMaximum(1080)
datumYtitle = QtWidgets.QLabel(MainWindow)
datumYtitle.setGeometry(70, 457, 80, 30)
font = QFont() 
font.setPointSize(18)
datumYtitle.setText("Y 軸")
datumYtitle.setFont(font)
datumYvalue = QtWidgets.QLabel(MainWindow)
datumYvalue.setGeometry(620, 457, 80, 30)
font = QFont() 
font.setPointSize(18)
datumYvalue.setText("555")
datumYvalue.setFont(font)
try:
    with open("setting.json", 'r') as file:
        data = json.load(file)
        datumXslider.setValue(data["middlePointX"])
        datumYslider.setValue(data["middlePointY"])
        datumXvalue.setText(str(data["middlePointX"]))
        datumYvalue.setText(str(data["middlePointY"]))
except FileNotFoundError:
    data = {"middlePointX": 0, "middlePointY": 0}
    datumXvalue.setText(str(0))
    datumYvalue.setText(str(0))
    with open("setting.json", 'w') as file:
        json.dump(data, file)
except Exception as e:
    print(e)

edge = {"offsetLeft": 0, "offsetRight": 0}

def datumXsliderValue(value):
    global data
    data["middlePointX"] = value
    datumXvalue.setText(str(data["middlePointX"]))
    # print("value: ", f"{value:03d}", end='\r')
datumXslider.valueChanged.connect(datumXsliderValue)

def datumYsliderValue(value):
    global data
    data["middlePointY"] = value
    datumYvalue.setText(str(data["middlePointY"]))
datumYslider.valueChanged.connect(datumYsliderValue)

ocv = True             # 設定全域變數，讓關閉視窗時 OpenCV 也會跟著關閉
def closeOpenCV():
    global ocv
    ocv = False        # 關閉視窗時，將 ocv 設為 False

MainWindow.closeEvent = closeOpenCV  # 設定關閉視窗的動作

def getEdge(pr):
    global data
    leftOffset = 0
    rightOffset = 0
    for x in range(data["middlePointX"]):
        if np.all(pr[data["middlePointX"]-x] == [128, 0, 0]):
            leftOffset += 1
        else:
            break
    prLong = len(pr)
    for x in range(prLong-data["middlePointX"]):
        if np.all(pr[x+data["middlePointX"]-1] == [128, 0, 0]):
            rightOffset += 1
        else:
            break
    edge["offsetLeft"] = leftOffset
    edge["offsetRight"] = rightOffset

def save():
    global data
    data = {"middlePointX": data["middlePointX"], "middlePointY": data["middlePointY"]}
    with open("setting.json", 'w') as file:
        json.dump(data, file)
    print("saved")
shortcut1 = QtWidgets.QShortcut(QKeySequence("Ctrl+S"), MainWindow)
shortcut1.activated.connect(save)


    # cv2.imshow("Predict ", predictImage)
class DeeplabV3(object):
    _defaults = {
        #-------------------------------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
        #-------------------------------------------------------------------#
        "model_path"        : '/Users/sam/Documents/MyProject/mixProject/TYAIcar/MLtraning/trainningCode/deeplabv3-plus-tf2-3.0/logs/ep084-loss0.216-val_loss0.116.h5',
        #----------------------------------------#
        #   所需要区分的类的个数+1
        #----------------------------------------#
        "num_classes"       : 7,
        #----------------------------------------#
        #   所使用的的主干网络：mobilenet、xception    
        #----------------------------------------#
        "backbone"          : "mobilenet",
        #----------------------------------------#
        #   输入图片的大小
        #----------------------------------------#
        "input_shape"       : [512, 512],
        #----------------------------------------#
        #   下采样的倍数，一般可选的为8和16
        #   与训练时设置的一样即可
        #----------------------------------------#
        "downsample_factor" : 16,
        #-------------------------------------------------#
        #   mix_type参数用于控制检测结果的可视化方式
        #
        #   mix_type = 0的时候代表原图与生成的图进行混合
        #   mix_type = 1的时候代表仅保留生成的图
        #   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
        #-------------------------------------------------#
        "mix_type"          : 0,
    }

    #---------------------------------------------------#
    #   初始化Deeplab
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #   获得模型
        #---------------------------------------------------#
        self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        self.model = Deeplabv3([self.input_shape[0], self.input_shape[1], 3], self.num_classes,
                                backbone = self.backbone, downsample_factor = self.downsample_factor)

        self.model.load_weights(self.model_path)
        # coremlModel = coremltools.convert(self.model)
        # coremlModel = coremltools.convert(self.model, 
        #                                   inputs=[coremltools.ImageType(shape=(1, 512, 512, 3), color_layout=coremltools.colorlayout.RGB)])
        # coremlModel = coremltools.convert(
        #     self.model,
        #     inputs=[coremltools.ImageType(
        #         # name='image',
        #         shape=(1, self.input_shape[0], self.input_shape[1], 3), 
        #         scale=1/255.0, 
        #         # bias=[-1,-1,-1], 
        #         color_layout='RGB'
        #     )]
        # )
        # coremlModel.save("TYAIroadModel_pixelBuffer.mlpackage")
        print('{} model loaded.'.format(self.model_path))
        
    @tf.function
    def get_pred(self, image_data):
        pr = self.model(image_data, training=False)
        # print(type(pr))
        return pr
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别 
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        #---------------------------------------------------------#
        #   归一化+通道数调整到第一维度+添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        #---------------------------------------------------#
        #   图片传入网络进行预测
        #---------------------------------------------------#
        pr = self.get_pred(image_data)[0].numpy()
        # print(type(pr))
        #---------------------------------------------------#
        #   将灰条部分截取掉
        #---------------------------------------------------#
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        #---------------------------------------------------#
        #   进行图片的resize
        #---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        #---------------------------------------------------#
        #   取出每一个像素点的种类
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1)
        # thread = threading.Thread(processPredict(pr, orininal_w, orininal_h))
        # thread.start()
        # processPredict(pr, orininal_w, orininal_h)


        
        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))
            getEdge(seg_img[data["middlePointY"]])
            #------------------------------------------------#
            #   将新图与原图及进行混合
            #------------------------------------------------#
            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))

        return image

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
deeplab = DeeplabV3()

video_path      = "/Users/sam/Documents/MyProject/mixProject/TYAIcar/MLtraning/visualIdentityVideo/IMG_9590.MOV"
video_save_path = ""
video_fps       = 60.0

def opencv():
    global ocv,video_path
    
    capture=cv2.VideoCapture(video_path)
    if video_save_path!="":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    ref, frame = capture.read()
    if not ref:
        raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

    fps = 0.0
    while(ocv):
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        if not ref:
            break
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        frame = np.array(deeplab.detect_image(frame))
        frame = cv2.circle(frame, (data["middlePointX"], data["middlePointY"]), radius=5, color=(255,255,255), thickness=10)
        frame = cv2.line(frame , (data["middlePointX"], data["middlePointY"]), (data["middlePointX"]-edge["offsetLeft"], data["middlePointY"]), (255,255,255), 4)
        frame = cv2.line(frame , (data["middlePointX"], data["middlePointY"]), (data["middlePointX"]+edge["offsetRight"], data["middlePointY"]), (255,255,255), 4)
        frame = cv2.resize(frame, (720, 405))
        height, width, channel = frame.shape
        bytesPerline = channel * width

        img = QImage(frame, width, height, bytesPerline, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img))
        
        # # RGBtoBGR满足opencv显示格式
        # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        
        # fps  = ( fps + (1./(time.time()-t1)) ) / 2
        # # print("fps= %.2f"%(fps))
        # frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # cv2.imshow("video",frame)
        c= cv2.waitKey(1) & 0xff 
        if video_save_path!="":
            out.write(frame)

        if c==27:
            capture.release()
            break

video = threading.Thread(target=opencv)
video.start()

MainWindow.show()
sys.exit(app.exec_())

# print("Video Detection Done!")
# capture.release()
# if video_save_path!="":
#     print("Save processed video to the path :" + video_save_path)
#     out.release()
# cv2.destroyAllWindows()
