from io import BytesIO
import cv2
import numpy as np
from PIL import Image

import time, sys, json
import threading

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *

from keras.models import load_model


import math

import serial
import samAi

from samAi import colors,deeplab
from serial import openSerial



# openCV

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
MainWindow.setObjectName("MainWindow")
MainWindow.setWindowTitle("TYAI car")
MainWindow.resize(864, 550)

label = QtWidgets.QLabel(MainWindow)
label.setGeometry(0, 0, 864, 480)

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


# 辨識

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



# run mode

#video_path = r"D:\Data\project\tyaiCar\TyaiCarSystem\IMG_1319.MOV"
video_path = r"D:\Data\project\tyaiCar\TyaiCarSystem\Test5.mp4"
#video_path = r"/Volumes/YihuanMiSSD/test8.MOV"

video_save_path = ""
video_fps = 30.0

useCam = False
CamID = 0


# main

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
    
    
    height, width, channel = 480, 864, 3


    fps = 0.0
    while True:
        t1 = time.time()

        # speed up
        for i in range(5):
            ref, frame = capture.read()

        frame = cv2.resize(frame, (864, 480))

        if not ref:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))

        # samAi
        deeplab.mix_type = 0
        result_img_blend, rd,result_img_trapezoid2 , modelOutput = deeplab.detect_image(frame)
        frame_blend = cv2.resize(np.array(result_img_blend), (width, height))

        angle = 90

        # get edge

        # ySampling = [260,300,340,380,420,460]
        # xSampling = [0,50,100,150,713,763,813,863]

        # roadPoints = []
        # roadPointsl = []
        # roadPointsr = []

        # for y in ySampling:
        #     resultPoint = getEdge(modelOutput[y])

        #     if resultPoint == [0,0]:
        #         continue

        #     roadPointsl.append([resultPoint[0],y])
        #     roadPointsr.append([resultPoint[1],y])

        # frame_blend = cv2.polylines(frame_blend, [np.array(roadPointsr, np.int32)], isClosed=False, color=(255, 255, 255), thickness=10)
        # frame_blend = cv2.polylines(frame_blend, [np.array(roadPointsl, np.int32)], isClosed=False, color=(255, 255, 255), thickness=10)

        # 梯形校正
        result_img_trapezoid_corrected = perspective_correction(np.array(frame_blend))

        # 梯形校正結果
        result_img_trapezoid_corrected_pil = Image.fromarray(result_img_trapezoid_corrected)
        result_img_trapezoid_np = np.array(result_img_trapezoid_corrected_pil.resize((250, 160), Image.BICUBIC))

        bytesPerline_trapezoid = 3 * result_img_trapezoid_np.shape[1]
        result_img_trapezoid_qt = QImage(result_img_trapezoid_np.data, result_img_trapezoid_np.shape[1],
                                        result_img_trapezoid_np.shape[0], bytesPerline_trapezoid, QImage.Format_RGB888)
        trapezoid_label.setPixmap(QPixmap.fromImage(result_img_trapezoid_qt))

        result_img_trapezoid_corrected = result_img_trapezoid_corrected.astype('uint8')



        
        bytesPerline_blend = channel * width
        img_blend = QImage(frame_blend.data, width, height, bytesPerline_blend, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img_blend))
        

        # 數據顯示
        painter = QPainter(label.pixmap())
        font = QFont()
        font.setPointSize(15)
        painter.setFont(font)
        font.setWeight(QFont.Bold)  # 設定為粗體

        painter.setPen(QColor(255, 255, 255))  # 文字顏色，白色
        painter.drawText(20, 30, f"FPS: {fps}")
        painter.drawText(20, 60, f"priority: right")


        painter.drawText(20, 240, f"turnGain: {0}")
        painter.drawText(20, 270, f"rightOffset: {0}")
        painter.drawText(20, 300, f"leftOffset: {0}")



        # 混合結果
        painter.end()

        


        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        #print("fps= %.2f"%(fps), end='\r')

        if openSerial:
            global ser
            ser.write((str(int(angle))+'\n').encode())

        c= cv2.waitKey(1) & 0xff 

        if video_save_path!="":
            out.write(frame)

        if c==27:
            print('end')
            capture.release()
            break


video = threading.Thread(target=opencv)
video.start()

MainWindow.show()
sys.exit(app.exec_())