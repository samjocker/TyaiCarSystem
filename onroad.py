from io import BytesIO
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

import math, os
import serial


openSerial = False

def animate_rocket():
  distance_from_top = 20
  for i in range(20):
    print("\n" * distance_from_top)
    print("          /\        ")
    print("          ||        ")
    print("          ||        ")
    print("         /||\        ")
    time.sleep(0.2)
    os.system('clear')  
    distance_from_top -= 1
    if distance_from_top < 0:
      distance_from_top = 20

if openSerial:
    print("Wait connect")
    COM_PORT = '/dev/cu.usbmodem1101'
    BAUD_RATES = 9600
    ser = serial.Serial(COM_PORT, BAUD_RATES)
    print("Connect successfuly")
    symbols = ['⣾', '⣷', '⣯', '⣟', '⡿', '⢿', '⣻', '⣽']
    for i in range(20):
        text = "<"
        for j in range(i):
            text += symbols[i%8]
        for j in range(20-i):
            text += " "
        text += ">"
        print(text, end='\r')
        time.sleep(0.1)
    print("Auto Pilot start!!!")
    ser.write((str(90)).encode())
    print("servo 90 degress")
    # time.sleep(5)
    # print("servoFree!!!")

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
MainWindow.setObjectName("MainWindow")
MainWindow.setWindowTitle("TYAI car")
MainWindow.resize(864, 550)

label = QtWidgets.QLabel(MainWindow)
label.setGeometry(0, 0, 864, 480)

colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
          (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
          (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
          (128, 64, 12)]


def getEdge(pr):


    global data, colors
    leftOffset = 0
    rightOffset = 0

    xCount = 0
    startType = 'none'

    if np.all(pr[0] == list(colors[1])):
        startType = 'road'

    lastType = startType
    lastX = 0
    highRange = [0,0]
    currentRange = [0, 0]

    for x in range(len(pr)):
        nowType = 'none'
        
        if np.all(pr[x] == list(colors[1])):
            nowType = 'road'

        if nowType == 'road':
            if currentRange[1] == 0:
                currentRange[0] = x
            currentRange[1] = x

            if currentRange[1] - currentRange[0] > highRange[1] - highRange[0]:
                highRange = currentRange
        else:
            currentRange = [0, 0] 
    #print(f"最長範圍：{highRange}         ",end='\r')

    return highRange

def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min


class DeeplabV3(object):
    _defaults = {
        "model_path": 'model/3_2.h5',
        "num_classes": 7,
        "backbone": "mobilenet",
        "input_shape": [387, 688],
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

        seg_img3 = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        image3   = Image.fromarray(np.uint8(seg_img3))
        image3_pil = Image.fromarray(np.uint8(seg_img3))
        modelOutput = seg_img3
        # # Convert PIL Image to bitmap (BytesIO)
        # image3_bytesio = BytesIO()
        # image3_pil.save(image3_bytesio, format='BMP')
        # image3_bytesio.seek(0)



        return image, pr, image2, modelOutput


deeplab = DeeplabV3()

#video_path = r"D:\Data\project\tyaiCar\TyaiCarSystem\IMG_1319.MOV"
video_path = r"/Volumes/YihuanMiSSD/test8.MOV"

video_save_path = ""
video_fps = 30.0

def calculate_angle(point1, point2):
    # point1 和 point2 是包含兩個座標值的元組 (x, y)
    
    # 計算差值
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]

    # 計算反正切值，注意要將結果轉換為度數
    angle_rad = math.atan2(delta_y, delta_x)
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def perspective_correction(image):



    # 定義原始四邊形的四個點
    original_points = np.float32([[-400, 480], [1120, 480],[200, 280], [520, 280]])

    # 定義梯形校正後的四個點
    corrected_points = np.float32([[200, 480], [520, 480], [200, 0], [520, 0]])

    # 計算透視變換矩陣
    perspective_matrix = cv2.getPerspectiveTransform(original_points, corrected_points)

    # 執行透視變換
    result = cv2.warpPerspective(image, perspective_matrix, (720, 480))

    return result


datumYslider = QtWidgets.QSlider(MainWindow)
datumYslider.setGeometry(0,490, 864, 30)
datumYslider.setOrientation(QtCore.Qt.Horizontal)
datumYslider.setMaximum(864)
datumYslider.setMinimum(0)
datumYslider.setValue(432)


trapezoid_label = QtWidgets.QLabel(MainWindow)
trapezoid_label.setGeometry(550, 0, 250, 160)
trapezoid_label.setStyleSheet("QLabel { background-color : white; color : black; }")
trapezoid_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

useCam = False
CamID = 0

from keras.models import load_model
result = 0

def opencv():
    
    if useCam:
        capture = cv2.VideoCapture(CamID)
    else:
        capture = cv2.VideoCapture(video_path)

    #to 480p
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if video_save_path != "":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    ref, frame = capture.read()
    if not ref:
        raise ValueError("Video source Error")

    cc=0

    fps = 0.0
    while True:
        t1 = time.time()
        for i in range(20):
            ref, frame = capture.read()

        frame = cv2.resize(frame, (864, 480))


        if not ref:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))

        deeplab.mix_type = 0
        result_img_blend, rd,result_img_trapezoid2 , modelOutput = deeplab.detect_image(frame)

        #print(modelOutput[400])
        height, width, channel = 480, 864, 3
        frame_blend = cv2.resize(np.array(result_img_blend), (width, height))


        offsetSum = 0
        offsetList = []
        takePoint = [290,305,320,335,350,365,380]
        lastx0 = 0
        lastx1 = 0
        lastTy = 0

        rightOffsetList = []
        leftOffsetList = []
        rightSidePoint = []
        leftSidePoint = []

        for Ty in takePoint:

            x0,x1 = getEdge(modelOutput[Ty])

            rightOffsetList.append(x1)
            leftOffsetList.append(x0)

            # if abs(x0 - lastx0) < 10:
            #     leftSidePoint.append(x0)
            
            # if abs(x1 - lastx1) < 10:
            #     rightSidePoint.append(x1)
            

            frame_blend = cv2.circle(frame_blend, (x0,Ty), radius=5, color=(0, 255,0))
            frame_blend = cv2.circle(frame_blend, (x1,Ty), radius=5, color=(0, 255,0))
            #cv2.putText(frame_blend, f"{x1 - x0}", (int((x0 + x1) / 2), Ty-15), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255, 0, 0), thickness=1)
            offsetSum += int((x0 + x1) / 2)
            offsetList.append(int((x0 + x1) / 2))

            if lastx0 != 0 and lastx1 != 0 and lastTy != 0:
                frame_blend = cv2.line(frame_blend, (lastx0,lastTy), (x0,Ty), (0, 255, 0), 2)
                frame_blend = cv2.line(frame_blend, (lastx1,lastTy), (x1,Ty), (0, 255, 0), 2)

            lastx0 = x0
            lastx1 = x1
            lastTy = Ty


        # roadType = "straight"
        # # 2,3,4的平均值大於0,5

        # rightUp = sum(rightOffsetList[2:5])/3 > rightOffsetList[5]
        # rightDown = sum(rightOffsetList[2:5])/3 > rightOffsetList[0]

        # leftUp = sum(leftOffsetList[2:5])/3 < leftOffsetList[5]
        # leftDown = sum(leftOffsetList[2:5])/3 < leftOffsetList[0]

        # if rightUp and rightDown and leftUp and leftDown:

        #     roadType = 'crossroads'

        # elif rightUp and rightDown and not leftUp and not leftDown:

        #     roadType = 'right fork road'
        
        # elif not rightUp and not rightDown and leftUp and leftDown:

        #     roadType = 'left fork road'
        
        # elif rightUp and not rightDown and leftUp and not leftDown:

        #     roadType = 'T-intersection'



        tv = 0
        if 864 - sum(rightOffsetList)/len(rightOffsetList) <60:
            tv = 864 - sum(rightOffsetList)/len(rightOffsetList)
        if sum(leftOffsetList)/len(leftOffsetList) <60:
            tv = -sum(leftOffsetList)/len(leftOffsetList)

        tv = int(tv)

        #offset = int((sum(rightOffsetList)/len(rightOffsetList) + sum(leftOffsetList)/len(leftOffsetList))/2)+ int(tv)

        offset = int((sum(rightOffsetList)/len(rightOffsetList))) -150

        #offset = int(sum(rightOffsetList)/len(rightOffsetList))-25
                
        runMode = 'drive on right'

        xSet = datumYslider.value() -432

        frame_blend = cv2.line(frame_blend, (offset+xSet+40,300), ( 432+120 ,450), (255, 255, 255), 2)
        frame_blend = cv2.line(frame_blend, (offset+xSet-40,300), ( 432-120 ,450), (255, 255, 255), 2)


        #角度計算
        angle = calculate_angle((offset+xSet,380), ( 432 ,450))

        cv2.putText(frame_blend, f"{int(angle)}", (360,440), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=2)



        bytesPerline_blend = channel * width
        img_blend = QImage(frame_blend.data, width, height, bytesPerline_blend, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img_blend))
        
        # 數據顯示
        painter = QPainter(label.pixmap())
        font = QFont()
        font.setPointSize(15)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))  # 文字顏色，白色
        painter.drawText(20, 30, f"FPS: {fps}")

        painter.drawText(20, 60, f"Mode: {runMode}")
        #painter.drawText(20, 90, f"RoadType: {roadType}")

        painter.drawText(20, 180, f"turnV: {tv}")
        painter.drawText(20, 210, f"rightOffset: {int(sum(rightOffsetList)/len(rightOffsetList))}")
        painter.drawText(20, 240, f"leftOffset: {int(sum(leftOffsetList)/len(leftOffsetList))}")

        painter.setPen(QColor(0, 255,0))  # 文字顏色，白色
        painter.drawText(300, 30, f"Auto Pilot ON")


        # 混合結果
        painter.end()

        # 梯形校正
        result_img_trapezoid_corrected = perspective_correction(np.array(result_img_blend))

        # 梯形校正結果
        result_img_trapezoid_corrected_pil = Image.fromarray(result_img_trapezoid_corrected)
        result_img_trapezoid_np = np.array(result_img_trapezoid_corrected_pil.resize((250, 160), Image.BICUBIC))

        bytesPerline_trapezoid = 3 * result_img_trapezoid_np.shape[1]
        result_img_trapezoid_qt = QImage(result_img_trapezoid_np.data, result_img_trapezoid_np.shape[1],
                                        result_img_trapezoid_np.shape[0], bytesPerline_trapezoid, QImage.Format_RGB888)
        trapezoid_label.setPixmap(QPixmap.fromImage(result_img_trapezoid_qt))



        output_folder_path = "road_corrected_images"

        # 確保資料夾存在，如果不存在就創建
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        # 指定檔案名稱，這裡使用固定的名稱，你也可以根據需要更改
        output_file_name = f"images{cc}.jpg"
        cc += 1

        # 組合完整的檔案路徑
        output_file_path = os.path.join(output_folder_path, output_file_name)

        # 將校正後的梯形影像儲存到資料夾
        result_img_trapezoid_corrected_pil.save(output_file_path)




        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        #print("fps= %.2f"%(fps), end='\r')

        if openSerial:
            global ser
            ser.write((str(int(angle))+'\n').encode())

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
