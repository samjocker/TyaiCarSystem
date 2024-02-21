import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from datetime import datetime
import os

import colorsys
import copy
import time, sys, json
import threading
import serial
import serial.tools.list_ports

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *

from nets.deeplab import Deeplabv3
from utils.utils import cvtColor, preprocess_input, resize_image

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import requests

import simpleaudio as sa

import xml.etree.ElementTree as ET

autoPilot = False

develeperMode = True

openSerial = True
cameraUse = True
mapControl = False

controlAngle = 90     

if openSerial:
    print("Wait connect")
    open_ports = []
    arduinoPorts = ""
    for port in serial.tools.list_ports.comports():
        try:
            ser = serial.Serial(port.device)
            ser.close()
            open_ports.append(port.device)
            if "/dev/cu.usb" in port.device:
                arduinoPorts = port.device
        except serial.SerialException:
            pass
    # COM_PORT = '/dev/cu.usbmodem13101'
    COM_PORT = arduinoPorts
    BAUD_RATES = 9600
    ser = serial.Serial(COM_PORT, BAUD_RATES)
    ser.timeout = 2
    _ = ser.read_all()
    print("Connect successfuly")
    time.sleep(2)
    print("start!")

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
MainWindow.setObjectName("MainWindow")
MainWindow.setWindowTitle("TYAI car")
MainWindow.resize(1220, 685)

label = QtWidgets.QLabel(MainWindow)
label.setGeometry(0, 0, 16*45, 9*45)

mapLabel = QtWidgets.QLabel(MainWindow)
mapLabel.setGeometry(16*45, 0, 500, 500)

y = 530
fpsText = QtWidgets.QLabel(MainWindow)
fpsText.setGeometry(70, y, 120, 30)
font = QFont() 
font.setPointSize(24)
fpsText.setFont(font)
fpsText.setText("Fps: "+str(0))

angleText = QtWidgets.QLabel(MainWindow)
angleText.setGeometry(180, y, 160, 30)
font = QFont() 
font.setPointSize(24)
angleText.setFont(font)
angleText.setText("Angle: "+str(0))

rawAngleText = QtWidgets.QLabel(MainWindow)
rawAngleText.setGeometry(300, y, 200, 30)
font = QFont() 
font.setPointSize(24)
rawAngleText.setFont(font)
rawAngleText.setText("Angle(raw): "+str(0))

speedText = QtWidgets.QLabel(MainWindow)
speedText.setGeometry(480, y, 100, 30)
font = QFont() 
font.setPointSize(24)
speedText.setFont(font)
speedText.setText("km/h: "+str(0))

screenSpeed = QtWidgets.QLabel(label)
screenSpeed.setGeometry(16*21, 15, 100, 60)
# font color should be white and 60 px
font = QFont()
font.setPointSize(60)
font.setBold(True)
font.setWeight(75)
screenSpeed.setFont(font)
screenSpeed.setText("0")
screenSpeed.setStyleSheet("color: white")

y += 50
rectWidthValue = QtWidgets.QSlider(MainWindow)
rectWidthValue.setGeometry(110, y, 500, 30)
rectWidthValue.setOrientation(QtCore.Qt.Horizontal)
rectWidthValue.setMaximum(1919)
rectWidthTitle = QtWidgets.QLabel(MainWindow)
rectWidthTitle.setGeometry(70, y-3, 80, 30)
font = QFont() 
font.setPointSize(18)
rectWidthTitle.setText("方寬")
rectWidthTitle.setFont(font)
rectWidthNum = QtWidgets.QLabel(MainWindow)
rectWidthNum.setGeometry(620, y-3, 80, 30)
font = QFont() 
font.setPointSize(18)
rectWidthNum.setText("555")
rectWidthNum.setFont(font)
y += 30
rectAdjustValue = QtWidgets.QSlider(MainWindow)
rectAdjustValue.setGeometry(110, y, 500, 30)
rectAdjustValue.setOrientation(QtCore.Qt.Horizontal)
rectAdjustValue.setMaximum(200)
rectAdjustTitle = QtWidgets.QLabel(MainWindow)
rectAdjustTitle.setGeometry(70, y-3, 80, 30)
font = QFont() 
font.setPointSize(18)
rectAdjustTitle.setText("微調")
rectAdjustTitle.setFont(font)
rectAdjustNum = QtWidgets.QLabel(MainWindow)
rectAdjustNum.setGeometry(620, y-3, 80, 30)
font = QFont() 
font.setPointSize(18)
rectAdjustNum.setText("555")
rectAdjustNum.setFont(font)
y += 30
siteValue = QtWidgets.QSlider(MainWindow)
siteValue.setGeometry(110, y, 500, 30)
siteValue.setOrientation(QtCore.Qt.Horizontal)
siteValue.setMaximum(4)
siteValue.setTickPosition(3)
siteValue.setValue(2)
# siteValue.setTickInterval(20)
siteTitle = QtWidgets.QLabel(MainWindow)
siteTitle.setGeometry(70, y-3, 80, 30)
font = QFont() 
font.setPointSize(18)
siteTitle.setText("  左")
siteTitle.setFont(font)
siteNum = QtWidgets.QLabel(MainWindow)
siteNum.setGeometry(620, y-3, 80, 30)
font = QFont() 
font.setPointSize(18)
siteNum.setText("右")
siteNum.setFont(font)

site = 2

colors = [ (150, 150, 150), (255, 255, 255), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]

try:
    with open("setting.json", 'r') as file:
        data = json.load(file)
        rectWidthValue.setValue(data["rectWidth"])
        rectAdjustValue.setValue(data["rectAdjust"])
        rectWidthNum.setText(str(data["rectWidth"]))
        rectAdjustNum.setText(str(data["rectAdjust"]))
except FileNotFoundError:
    data = {"middlePointX": 0, "middlePointY": 0, "sideDistValue":0, "trapezoidXvalue": 0, "trapezoidYvalue": 0, "rectWidth": 0, "rectAdjust": 0}
    rectWidthNum.setText(str(0))
    rectAdjustNum.setText(str(0))
    with open("setting.json", 'w') as file:
        json.dump(data, file)
except KeyError:
    data = {"middlePointX": 0, "middlePointY": 0, "sideDistValue":0, "trapezoidXvalue": 0, "trapezoidYvalue": 0, "rectWidth": 0, "rectAdjust": 0}
    rectWidthNum.setText(str(0))
    rectAdjustNum.setText(str(0))
    with open("setting.json", 'w') as file:
        json.dump(data, file)
except Exception as e:
    print(e)

def play_sound(sound_file, end=False):
    wave_obj = sa.WaveObject.from_wave_file(sound_file)
    play_obj = wave_obj.play()
    if end:
        play_obj.wait_done()

def rectWidthChange(value):
    data["rectWidth"] = value
    rectWidthNum.setText(str(value))
rectWidthValue.valueChanged.connect(rectWidthChange)
def rectAdjustChange(value):
    data["rectAdjust"] = value
    rectAdjustNum.setText(str(value))
rectAdjustValue.valueChanged.connect(rectAdjustChange)
def siteChange(value):
    global site
    site = value
siteValue.valueChanged.connect(siteChange)

ocv = True            

def save():
    # data = {"middlePointX": data["middlePointX"], "middlePointY": data["middlePointY"]}
    with open("setting.json", 'w') as file:
        json.dump(data, file)
    print("saved")
shortcut1 = QtWidgets.QShortcut(QKeySequence("Ctrl+S"), MainWindow)
shortcut1.activated.connect(save)

lastPlay = time.time()
def playWarning():
    global lastPlay
    if time.time() - lastPlay >= 1.5:
        lastPlay = time.time()
        play_sound("sound/warning.wav")

def map(value, from_low, from_high, to_low, to_high):
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low

def mask_image(imgdata, angle, size = 150): 
    global data
    # Load image 
    # image = QImage.fromData(imgdata, imgtype) 
    h,w,c = imgdata.shape
    image = QImage(imgdata.data, w, h, 3*w, QImage.Format_RGB888)
  
    # convert image to 32-bit ARGB (adds an alpha 
    # channel ie transparency factor): 
    image.convertToFormat(QImage.Format_ARGB32) 
  
    # Crop image to a square: 
    imgsize = min(image.width(), image.height()) 
    rect = QtCore.QRect( 
        (image.width() - imgsize) / 2, 
        (image.height() - imgsize) / 2, 
        imgsize, 
        imgsize, 
     ) 
       
    image = image.copy(rect) 
  
    # Create the output image with the same dimensions  
    # and an alpha channel and make it completely transparent: 
    out_img = QImage(imgsize, imgsize, QImage.Format_ARGB32) 
    out_img.fill(QtCore.Qt.transparent) 
  
    # Create a texture brush and paint a circle  
    # with the original image onto the output image: 
    brush = QBrush(image) 
  
    # Paint the output image 
    painter = QPainter(out_img) 
    painter.setBrush(brush) 
  
    # Don't draw an outline 
    painter.setPen(QtCore.Qt.NoPen) 
  
    # drawing circle 
    painter.drawEllipse(0, 0, imgsize, imgsize) 
  
    # closing painter event 
    painter.end() 
  
    # Convert the image to a pixmap and rescale it.  
    pr = QWindow().devicePixelRatio() 
    pm = QPixmap.fromImage(out_img) 
    pm.setDevicePixelRatio(pr) 
    size *= pr 
    pm = pm.scaled(size, size, QtCore.Qt.KeepAspectRatio,  
                               QtCore.Qt.SmoothTransformation) 
  
    # Rotate the pixmap by 30 degrees
    pm = pm.transformed(QTransform().rotate(120))
    # Return back the rotated pixmap data
    return pm

lastTime = time.time()
lastRoute = []
speed = 150
slidingJsonData = {}
def slidingWindow(frame):
    global data, colors, site, openSerial, lastTime, autoPilot, lastRoute, rawAngleText, speed, slidingJsonData  
    rectColor = (0, 200 ,0)

    cdnY = 1079
    rectHeight = 30
    rectWidth = int(data["rectWidth"])
    adjustNum = data["rectAdjust"]
    suggestSite = True
    points = []
    # [959, 1079]
    # TODO:add turn site identify
    while (cdnY-rectHeight >= 0):
        x1 = int((1919-rectWidth)/2)
        if rectWidth > 20:
            block = [[x1,x1+int(rectWidth/2)], [x1+int(rectWidth/2), x1+rectWidth]]
            blockPercent = [0, 0]
            keepAdjust = True
            runTime = 0
            lastBlock = []
            addNum = 20
            biggest = {"cdn": [], "dist": 50}
            testPoints = []
            while keepAdjust:
                block = [[x1,x1+int(rectWidth/2)], [x1+int(rectWidth/2), x1+rectWidth]]
                for b in range(2):
                    start_row, end_row = cdnY-rectHeight, cdnY
                    start_col, end_col = block[b][0], block[b][1]
                    roi = frame[start_row:end_row, start_col:end_col]
                    target_color = np.array([colors[1][2], colors[1][1], colors[1][0]], dtype=np.uint8)
                    if roi.size == 0:
                        pass
                        # print("區域大小為零，無法計算佔比。")
                    else:
                        mask = cv2.inRange(roi, target_color, target_color)
                        total_pixels = np.count_nonzero(mask)
                        if total_pixels != 0:
                            percentage = int((total_pixels / ((roi.shape[0]-3) *( roi.shape[1]-3))) * 100)
                            blockPercent[b] = percentage

                runTime += 1

                if site == 0 or site == 1:
                    # addNum = -20
                    if block[0][0] < 0:
                        if lastBlock != []:
                            keepAdjust = False
                            points.append([lastBlock[0][1], cdnY-rectHeight])
                            block = lastBlock
                            break
                        else:
                            # x1 = int((1919-rectWidth)/2)
                            addNum = abs(addNum)*-1
                    elif block[1][1] > 1919:
                        if lastBlock != []:
                            keepAdjust = False
                            points.append([lastBlock[0][1], cdnY-rectHeight])
                            break
                        elif biggest["cdn"] != []:
                            keepAdjust = False
                            points.append([biggest["cdn"][0][1], cdnY-rectHeight])
                            block = biggest["cdn"]
                            break
                        else:
                            break
                    elif abs(blockPercent[0]-blockPercent[1]) <= 20 and blockPercent[1] > 40:
                        lastBlock = block
                        if abs(blockPercent[0]-blockPercent[1]) < biggest["dist"]:
                            if biggest["cdn"] != []:
                                if block[1][0] < biggest["cdn"][1][0]:
                                    biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                                    biggest["cdn"] = block
                            else:
                                biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                                biggest["cdn"] = block
                    else:
                        if lastBlock != []:
                            keepAdjust = False
                            block = lastBlock
                            points.append([block[0][1], cdnY-rectHeight])
                            break
                        else:
                            if abs(blockPercent[0]-blockPercent[1]) < biggest["dist"] and blockPercent[1] > 20 :
                                if biggest["cdn"] != []:
                                    if block[1][0] < biggest["cdn"][1][0]:
                                        biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                                        biggest["cdn"] = block
                                else:
                                    biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                                    biggest["cdn"] = block
                    x1 -= addNum

                elif site == 2:
                    if abs(blockPercent[0]-blockPercent[1]) <= 30 and blockPercent[0] > 5:
                        points.append([block[0][1], cdnY-rectHeight])
                        keepAdjust = False
                        break
                    elif block[0][0] < 0 or block[1][1] > 1919:
                        # points.append([959, cdnY-rectHeight])
                        keepAdjust = False
                        break
                    else:
                        if blockPercent[0] > blockPercent[1] and blockPercent[0] > 10:
                            suggestSite = False
                            addNum = abs(addNum)*-1
                        elif blockPercent[0] < blockPercent[1] and blockPercent[0] > 10:
                            suggestSite = True
                            addNum = abs(addNum)
                        else:
                            if suggestSite:
                                addNum = abs(addNum)
                            else:
                                addNum = abs(addNum)*-1
                        x1 += addNum

                elif site ==3 or site == 4:
                    # addNum = 20
                    if block[1][1] > 1919:
                        if lastBlock != []:
                            keepAdjust = False
                            points.append([lastBlock[0][1], cdnY-rectHeight])
                            block = lastBlock
                            break
                        else:
                            # x1 = int((1919-rectWidth)/2)
                            # x1 = 960
                            addNum = abs(addNum)*-1
                    elif block[0][0] < 0:
                        # print("\nout\n")
                        if lastBlock != []:
                            keepAdjust = False
                            points.append([lastBlock[0][1], cdnY-rectHeight])
                            break
                        elif biggest["cdn"] != []:
                            keepAdjust = False
                            points.append([biggest["cdn"][0][1], cdnY-rectHeight])
                            block = biggest["cdn"]
                            break
                        else:
                            break
                    elif abs(blockPercent[0]-blockPercent[1]) <= 20 and blockPercent[0] > 40:
                        lastBlock = block
                        if abs(blockPercent[0]-blockPercent[1]) < biggest["dist"]:
                            if biggest["cdn"] != []:
                                if block[1][0] > biggest["cdn"][1][0]:
                                    biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                                    biggest["cdn"] = block
                            else:
                                biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                                biggest["cdn"] = block
                    else:
                        if lastBlock != []:
                            keepAdjust = False
                            block = lastBlock
                            points.append([block[0][1], cdnY-rectHeight])
                            break
                        else:
                            if abs(blockPercent[0]-blockPercent[1]) < biggest["dist"] and blockPercent[0] > 20:
                                if biggest["cdn"] != []:
                                    if block[1][0] > biggest["cdn"][1][0]:
                                        biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                                        biggest["cdn"] = block
                                else:
                                    biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                                    biggest["cdn"] = block
                    x1 += addNum
                # testPoints.append([block[0][1], cdnY-rectHeight])
                if runTime >= 100:
                    keepAdjust = False
                    # print(blockPercent)
                    print("break", blockPercent[0], blockPercent[1], site)
            # testPoints_array = np.array(testPoints, dtype=np.int32)
            # cv2.polylines(frame, [testPoints_array], isClosed=False, color=(235, 0, 0), thickness=20)
            if autoPilot and develeperMode:
                cv2.rectangle(frame, (block[0][0], cdnY-rectHeight), (block[1][1], cdnY), rectColor, 4, cv2.LINE_AA)
        #     cv2.putText(frame, str(blockPercent[0]), (block[0][0]-130, cdnY-10), cv2.FONT_HERSHEY_SIMPLEX,
        # 2, (0, 255, 255), 4, cv2.LINE_AA)
        #     cv2.putText(frame, str(blockPercent[1]), (block[1][1]+10, cdnY-10), cv2.FONT_HERSHEY_SIMPLEX,
        # 2, (0, 255, 255), 4, cv2.LINE_AA)
        cdnY -= rectHeight
        rectWidth -= adjustNum
        rectWidth = max(rectWidth, 0)

    fps = round(1.0/(time.time()-lastTime), 2)
    lastTime = time.time()
    fpsText.setText("Fps: "+str(fps))
    
    if len(points) > 8:
        points_array = np.array(points[:-1], dtype=np.int32)
        if site == 0 or site == 4:
            points_array = points_array[:11]
        elif site == 2:
            points_array = points_array[:11]

        if develeperMode or autoPilot:
            cv2.polylines(frame, [points_array], isClosed=False, color=(100, 157 ,236), thickness=40)
        if blockPercent[0]+blockPercent[1] < 10:
            rectColor = (200, 0, 0)
        coords = points_array
        if site == 0 or site == 4:
            median_coords = np.median(coords, axis=0)
        elif site == 1 or site == 3:
            median_coords = np.mean(coords, axis=0)
            points_arrayNum = len(points_array)
            # print(points_arrayNum)
            if points_arrayNum > 8:
                subLine1 = np.mean(points_array[:int(points_arrayNum/2)], axis=0)
                subLine2 = np.mean(points_array[int(points_arrayNum/2+1):int(points_arrayNum-3)], axis=0)
                line1Angle = np.arctan2(points_array[0][1]-points_array[int(points_arrayNum/2)][1], points_array[0][0]-points_array[int(points_arrayNum/2)][0])
                line1Angle = np.degrees(line1Angle)
                median_coords = np.mean([subLine1, subLine2], axis=0)
                if site == 1 and line1Angle < 55:
                    median_coords = np.mean([points_array[2], points_array[-3]], axis=0)
                    if develeperMode:
                        cv2.line(frame, (points_array[2][0], points_array[2][1]), (points_array[int(points_arrayNum/2)][0], points_array[int(points_arrayNum/2)][1]), (0, 0, 255), 40)
                        cv2.line(frame, (points_array[int(points_arrayNum/2+1)][0], points_array[int(points_arrayNum/2)+1][1]), (points_array[-3][0], points_array[-3][1]), (0, 0, 255), 40)
                elif site == 1 and line1Angle >= 55:
                    median_coords = subLine1
                elif site == 3 and line1Angle > 125:
                    median_coords = np.mean([points_array[2], points_array[-3]], axis=0)
                    if develeperMode:
                        cv2.line(frame, (points_array[2][0], points_array[2][1]), (points_array[int(points_arrayNum/2)][0], points_array[int(points_arrayNum/2)][1]), (0, 0, 255), 40)
                        cv2.line(frame, (points_array[int(points_arrayNum/2+1)][0], points_array[int(points_arrayNum/2)+1][1]), (points_array[-3][0], points_array[-3][1]), (0, 0, 255), 40)
                elif site == 3 and line1Angle <= 125:
                    median_coords = np.median(points_array[:int(points_arrayNum/2+4)], axis=0)
        else:
            maxCdn = np.max(coords, axis=0)
            minCdn = np.min(coords, axis=0)
            if abs(maxCdn[0]-959) > abs(minCdn[0]-959):
                median_coords = maxCdn
            else:
                median_coords = minCdn
        if site >= 3:
            point_coords = np.array([1200, 1079])
        elif site <= 1:
            point_coords = np.array([718, 1079])
        else:
            point_coords = np.array([959, 1079])
        if develeperMode:
            cv2.circle(frame, (int(median_coords[0]), int(median_coords[1])), 15, (181, 99, 235), -1)
            cv2.circle(frame, (int(point_coords[0]), int(point_coords[1])), 15, (0, 0, 255), -1)
        relative_coords = point_coords - median_coords
        angle_rad = np.arctan2(relative_coords[1], relative_coords[0])
        angle_deg = np.degrees(angle_rad)
        rawAngleText.setText("Angle(raw): "+str(int(angle_deg)))

        if site == 1 or site == 3:
            if angle_deg >= 130 or angle_deg <= 50:
                muiltNum = 1.1
            else:
                muiltNum = 0.8
        elif site == 2:
            muiltNum = 1.0
        elif site == 4:
            if angle_deg >= 130:
                muiltNum = 1.2
            elif angle_deg <= 80:
                muiltNum = 1.2
            else:
                muiltNum = 0.8
        elif site == 0:
            if angle_deg <= 50:
                muiltNum = 1.2
            elif angle_deg >= 100:
                muiltNum = 1.2
            else:
                muiltNum = 0.8
        angle_deg = int(max(min(90+(angle_deg-90)*muiltNum, 180), 0))

        angleText.setText("Angle: "+str(angle_deg))
        # print("fps= %.2f, angle= %4d"%(fps, angle_deg), end='\r')
        if angle_deg > 120 or angle_deg < 60:
            speed = 100
        else:
            speed = 150

        if openSerial:
            global ser, controlAngle
            ser.write((str(int(angle_deg))+'\n').encode())
            # send to arduino speedStr format is "40speed" ex: "4050" and "4100" mean 50 and 100 speed
            speedStr = "4"+str(speed).zfill(3)+"\n"
            # time.sleep(0.1)
            # print(speedStr)
            # time.sleep(0.02)
            # ser.write(speedStr.encode())

        # if openSerial:
        #     serData = ser.readline().decode()
        #     print("serData: "+serData)
        #     if serData == "apon\r\n":
        #         autoPilot = True
        #         play_sound("sound/autoPilotON.wav")
        #     elif serData == "apoff\r\n":
        #         autoPilot = False
        #         play_sound("sound/autoPilotOFF.wav")
        #     elif serData == "apfail\r\n":
        #         autoPilot = False
        #         play_sound("sound/autoPilotOFF.wav")

        slidingJsonData = {"angle": angle_deg, "points":points, "fps":fps, "mode":site, "APstate": autoPilot}
    else:
        playWarning()

    return frame
class DeeplabV3(object):
    _defaults = {
        "model_path"        : 'model/3_7_5.h5',
        "num_classes"       : 7,
        "backbone"          : "mobilenet",
        "input_shape"       : [387, 688],
        "downsample_factor" : 16,
        "mix_type"          : 1,
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
                                backbone = self.backbone, downsample_factor = self.downsample_factor)

        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))
        
    @tf.function
    def get_pred(self, image_data):
        pr = self.model(image_data, training=False)
        # print(type(pr))
        return pr
    def detect_image(self, image):
        image       = cvtColor(image)

        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]

        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        pr = self.get_pred(image_data)[0].numpy()
        # print(type(pr))

        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)

        pr = pr.argmax(axis=-1)
        # thread = threading.Thread(processPredict(pr, orininal_w, orininal_h))
        # thread.start()
        # processPredict(pr, orininal_w, orininal_h)


        
        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

            # kernel = np.ones((7,7),np.uint8)
            # seg_img = cv2.dilate(seg_img,kernel,iterations = 5)
            # seg_img = cv2.erode(seg_img,kernel,iterations = 5)
            slidingWindow(seg_img)
            # findLine(seg_img)
            image   = Image.fromarray(np.uint8(seg_img))

            image   = Image.blend(old_img, image, 0.5)

        elif self.mix_type == 1:
            
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            slidingWindow(seg_img)

            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')

            image = Image.fromarray(np.uint8(seg_img))

        return image

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
deeplab = DeeplabV3()

video_path      = "/Users/sam/Documents/MyProject/mixProject/TYAIcar/MLtraning/visualIdentityVideo/IMG_1420.MOV"
#filename format is time ex: 2023_01_12_13_23_30
fileName  = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
video_save_path = ""
video_fps       = 30.0

# mapImg = output = np.zeros((150, 150, 3), dtype="uint8")

json_Data = {}
def startMap():
    global mapLabel, siteValue, speedText, autoPilot, ser

    # 讀取OSM檔案
    osm_file_path = "test/TYAIcampus3.osm"
    G = ox.graph_from_point((24.99250, 121.32032), dist=200, network_type='drive_service')
    tree = ET.parse(osm_file_path)
    root = tree.getroot()

    # 創建節點字典以便根據ID查找座標
    node_dict = {}
    for node in root.findall('node'):
        node_id = node.attrib['id']
        latitude = float(node.attrib['lat'])
        longitude = float(node.attrib['lon'])
        node_dict[node_id] = (latitude, longitude)

    # 創建way字典以便根據ID查找節點參考和標籤
    way_dict = {}
    for way in root.findall('way'):
        way_id = way.attrib['id']
        node_refs = [nd.attrib['ref'] for nd in way.findall('nd')]
        tags = {tag.attrib['k']: tag.attrib['v'] for tag in way.findall('tag')}
        way_dict[way_id] = {'node_refs': node_refs, 'tags': tags}

    # 獲取路徑的範圍
    min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf

    # 遍歷所有way並僅顯示service等於driveway的路徑
    for way_id, data in way_dict.items():
        if 'service' in data['tags'] and data['tags']['service'] == 'driveway':
            route = []
            # 從way中提取節點座標
            for node_id in data['node_refs']:
                if node_id in node_dict:
                    route.append(node_dict[node_id])
            route = np.array(route)

            # 更新範圍
            min_x = min(min_x, np.min(route[:, 1]))
            max_x = max(max_x, np.max(route[:, 1]))
            min_y = min(min_y, np.min(route[:, 0]))
            max_y = max(max_y, np.max(route[:, 0]))

    # 計算縮放比例
    width = 500
    height = 500
    scale_x = width / (max_x - min_x)
    scale_y = height / (max_y - min_y)

    def convertCdn(num, type):
        if type == "x":
            return int((num - min_x) * scale_x)
        elif type == "y":
            return int((num - min_y) * scale_y)

    # 創建空白畫布
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img.fill(255)

    # 繪製每個way的路徑並顯示節點
    canGoWay = []
    for way_id, data in way_dict.items():
        if 'service' in data['tags'] and data['tags']['service'] == 'driveway':
            route = []
            # 從way中提取節點座標
            for node_id in data['node_refs']:
                if node_id in node_dict:
                    route.append(node_dict[node_id])
            route = np.array(route)
            
            # 繪製路徑
            for i in range(len(route) - 1):
                u = route[i]
                v = route[i + 1]
                ux = convertCdn(u[1], "x")
                uy = convertCdn(u[0], "y")
                vx = convertCdn(v[1], "x")
                vy = convertCdn(v[0], "y")
                cv2.line(img, (ux, uy), (vx, vy), (217, 232, 245), 8)
                try:
                    site = data['tags']['target']
                    canGoWay.append([(ux, uy), (vx, vy), site])
                except:
                    pass
            
            # 繪製節點
            for node in route:
                x = int((node[1] - min_x) * scale_x - 1)
                y = int((node[0] - min_y) * scale_y - 1)
                cv2.circle(img, (x, y), 3, (134, 171, 212), -1)
    startX, startY = 121.32167, 24.99179
    startX, startY = 121.32162, 24.99230
    endX, endY = 121.31989, 24.99412

    origin = ox.distance.nearest_nodes(G, startX, startY)
    destination = ox.distance.nearest_nodes(G, endX, endY)

    route = nx.shortest_path(G, origin, destination)
    lastNode = (convertCdn(startX, "x"), convertCdn(startY, "y"))
    routeList = []
    for node in route:
        x, y = G.nodes[node]['x'], G.nodes[node]['y']
        x = convertCdn(x, "x")
        y = convertCdn(y, "y")
        cv2.line(img, lastNode, (x, y), (48, 66, 105), 3)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        routeList.append((x, y))
        lastNode = (x, y)
        # cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
    cv2.line(img, lastNode, (convertCdn(endX, "x"), convertCdn(endY, "y")), (48, 66, 105), 3)
    cv2.circle(img, (convertCdn(startX, "x"), convertCdn(startY, "y")), 6, (242, 97, 1), -1)
    cv2.circle(img, (convertCdn(endX, "x"), convertCdn(endY, "y")), 4, (255, 255, 255), -1)
    cv2.circle(img, (convertCdn(endX, "x"), convertCdn(endY, "y")), 6, (242, 97, 1), 2)
    
    def get_coordinates():
        global json_Data, screenSpeed, autoPilot
        api_url = "https://6c96-60-251-221-219.ngrok-free.app/getData"
        response = requests.get(api_url)
        APvalue = 1 if autoPilot else 0
        APurl = "https://6c96-60-251-221-219.ngrok-free.app/update/"+str(APvalue)
        response = requests.get(APurl)
        data = response.json()
        # print(data)
        latitude = data["latitude"]
        longitude = data["longitude"]
        site = int(float(data["site"]))
        speed = int(float(data["speed"]))
        if speed >= 0:
            screenSpeed.setText(str(speed))
        else:
            screenSpeed.setText(str(0))
        json_Data = {}
        json_Data = {"latitude": latitude, "longitude": longitude, "site": site, "speed": speed}
        return latitude, longitude, site, speed
    
    def point_to_line_distance(point, line):
        """
        計算點到直線的垂直距離
        :param point: 點的座標 (x, y)
        :param line: 直線的兩個端點 ((x1, y1), (x2, y2))
        :return: 點到直線的垂直距離
        """
        x, y = point
        x1, y1 = line[0]
        x2, y2 = line[1]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            # 線段長度為0，返回點與端點之間的距離
            return np.linalg.norm(np.array(point) - np.array(line[0]))
        t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))  # 確保t在[0, 1]範圍內
        closest_point = (x1 + t * dx, y1 + t * dy)
        distance = np.linalg.norm(np.array(point) - np.array(closest_point))
        return distance
    
    lastImg = img.copy()
    turnSite = ""
    turnMode = "littleRight"
    turnAngle = 0
    readyToOut = False
    readyToIn = True
    while True:
        myCdn = [121.32186, 24.99240]
        siteAngle = 0       
        gpsSpeed = 0 
        try:
            myCdn[0], myCdn[1], siteAngle, gpsSpeed = get_coordinates()
            speedText.setText("km/h: "+str(gpsSpeed))
        except Exception as e:
            print(e)

        if openSerial:
            serData = ser.readline().decode()
            print(serData, end="\n\n")
            if serData == "apon\r\n":
                autoPilot = True
                play_sound("sound/autoPilotON.wav")
            elif serData == "apoff\r\n":
                autoPilot = False
                play_sound("sound/autoPilotOFF.wav")
            elif serData == "apoffw\r\n":
                autoPilot = False
                playWarning()
            elif serData == "apfail\r\n":
                autoPilot = False
                play_sound("sound/autoPilotOFF.wav")

        myCdn = [convertCdn(myCdn[0], "x"), convertCdn(myCdn[1], "y")]

        # 找出距離最近的線條
        closest_way = None
        closest_distance = np.inf
        for way in canGoWay:
            u, v, site = way
            dist = point_to_line_distance(myCdn, (u, v))
            if dist < closest_distance:
                closest_way = way
                closest_distance = dist

        # 打印最接近的線條和距離
        if closest_way is not None:
            u, v, recommandSite = closest_way
            # cv2.line(img, u, v, (0, 255, 0), 4)
            # print("Closest way:", u, "-", v, "Distance:", closest_distance, "Site:", recommandSite)
        else:
            print("No closest way found.")

        img = lastImg.copy()
        nowCdn = tuple(myCdn)
        cv2.circle(img, nowCdn, 4, (223, 251, 252), -1)
        cv2.circle(img, nowCdn, 6, (61, 91, 129), 2)
        distance = cv2.norm(nowCdn, routeList[0])
        # print("distance: ", distance)

        if len(routeList) > 2:
            distance2 = cv2.norm(nowCdn, routeList[1])
            if distance > distance2:
                cv2.circle(lastImg, routeList[0], 6, (0, 0, 255), -1)
                routeList.pop(0)
                print("delete one pass point")

        if distance < 25 and readyToIn and not readyToOut:
            readyToIn = False
            if len(routeList) > 2:
                turnAngle = np.arctan2(routeList[1][1]-routeList[0][1], routeList[1][0]-routeList[0][0])
                turnAngle = np.degrees(turnAngle)
                # print("turnAngle: ", turnAngle)
            elif routeList[0] == (178, 98) or routeList[0] == (168, 112):
                turnSite = ""
                recommandSite = "littleRight"
            elif turnAngle >= -45 and turnAngle < 45:
                turnSite = "east"
            elif turnAngle >=45 and turnAngle < 135:
                turnSite = "south"
            elif turnAngle >= 135 and turnAngle < 180 or turnAngle >= -180 and turnAngle < -135:
                turnSite = "west"
            else:
                turnSite = "north"
        elif distance < 10 and not readyToOut:
            readyToOut = True
        elif distance > 11 and readyToOut:
            readyToOut = False
            readyToIn = True
            turnSite = ""
            cv2.circle(lastImg, routeList[0], 6, (255, 0, 0), -1)
            if routeList[0] == (222, 73):
                recommandSite = "littleRight"
            routeList.pop(0)
            print("pass one point")

        if routeList[0] == (222, 73):
            turnSite = ""
            recommandSite = "straight"

        siteStr = ""
        if turnSite == "west" or turnSite == "east":
            if siteAngle >= 315 or siteAngle <= 45:
                siteStr = "north"
            elif siteAngle >= 135 and siteAngle <= 225:
                siteStr = "south"
            else:
                siteStr = ""
                recommandSite = "east" if siteAngle >= 45 and siteAngle <= 135 else "west"
        elif turnSite == "north" or turnSite == "south":
            if siteAngle >= 225 and siteAngle <= 315:
                siteStr = "west"
            elif siteAngle >= 45 and siteAngle <= 135:
                siteStr = "east"
            else:
                siteStr = ""
                recommandSite = "north" if siteAngle >= 315 and siteAngle <= 45 else "south"
        elif recommandSite == "east" or recommandSite == "west":
            if siteAngle >= 270 or siteAngle <= 90:
                siteStr = "north"
            else:
                siteStr = "south"
        elif recommandSite == "north" or recommandSite == "south":
            if siteAngle >= 0 and siteAngle <= 180:
                siteStr = "east"
            else:
                siteStr = "west"

        # turnMode have five step left, little left, straight, little right, right

        # turnMode = ""
        if turnSite != "":    
            if turnSite == "north":
                if siteStr == "east":
                    turnMode = "Left"
                elif siteStr == "west":
                    turnMode = "Right"
            elif turnSite == "east":
                if siteStr == "south":
                    turnMode = "Left"
                elif siteStr == "north":
                    turnMode = "Right"
            elif turnSite == "south":
                if siteStr == "west":
                    turnMode = "Left"
                elif siteStr == "east":
                    turnMode = "Right"
            elif turnSite == "west":
                if siteStr == "north":
                    turnMode = "Left"
                elif siteStr == "south":
                    turnMode = "Right"
        else:
            if recommandSite == "north":
                if siteStr == "east":
                    turnMode = "littleLeft"
                elif siteStr == "west":
                    turnMode = "littleRight"
            elif recommandSite == "east":
                if siteStr == "south":
                    turnMode = "littleLeft"
                elif siteStr == "north":
                    turnMode = "littleRight"
            elif recommandSite == "south":
                if siteStr == "west":
                    turnMode = "littleLeft"
                elif siteStr == "east":
                    turnMode = "littleRight"
            elif recommandSite == "west":
                if siteStr == "north":
                    turnMode = "littleLeft"
                elif siteStr == "south":
                    turnMode = "littleRight"
            elif recommandSite == "littleRight":
                turnMode = "littleRight"
            elif recommandSite == "straight":
                turnMode = "straight"

        # print("siteStr: ", siteStr, siteAngle)
        print("turnSite: ", turnSite, "turnMode: ", turnMode)
        # print(routeList)

        # 創建網絡圖
        for way_id, data in way_dict.items():
            for i in range(len(data['node_refs']) - 1):
                u = data['node_refs'][i]
                v = data['node_refs'][i + 1]
                G.add_edge(u, v)

        # nowCdn = (convertCdn(121.32164781214851, "x"), convertCdn(24.99193515292301, "y"))
        sideDict = {"Left": 0,"littleLeft": 1,"straight": 2,"littleRight": 3,"Right": 4}
        if mapControl:
            siteValue.setValue(sideDict[turnMode])
        image = QImage(cv2.flip(img, 0), 500, 500, 1500, QImage.Format_RGB888)
        mapLabel.setPixmap(QPixmap.fromImage(image))
        time.sleep(0.3)


folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = "testLog/"+folder_name
os.makedirs(folder_name)
frame_count = 0
frames_info = []

def opencv():
    global ocv,video_path,video_save_path,video_fps, sideButtonState, openSerial, frame_count, frames_info, folder_name, json_Data, slidingJsonData
    
    capture=cv2.VideoCapture(0 if cameraUse else video_path)
    if video_save_path!="":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    ref, frame = capture.read()
    if not ref:
        raise ValueError("Video source Error")

    while(ocv):
        for _ in range(1 if cameraUse else 9):
            ref, frame = capture.read()
            frame = cv2.resize(frame, (1920, 1080))
            if cameraUse:
            #     frame = cv2.flip(frame, 0)
                frame = cv2.flip(frame, 1)
            if not ref:
                ocv = False
                capture.release()
                break

        frame_name = f"{folder_name}/frame_{frame_count:04d}.jpg"
        cv2.imwrite(frame_name, frame)
        readyToWriteJson = {"file_name": frame_name, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}
        readyToWriteJson.update(json_Data)
        readyToWriteJson.update(slidingJsonData)

        if video_save_path!="" and cameraUse:
            out.write(frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        frame = np.array(deeplab.detect_image(frame))

        height, width, channel = 9*47, 16*47, 3
        frame = cv2.resize(frame, (width, height))
        bytesPerline = channel * width

        img = QImage(frame, width, height, bytesPerline, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img))
        # print("fps= %.2f, angle= %4d"%(fps, 90), end='\r')

        frames_info.append(readyToWriteJson)
        frame_count += 1

        json_data = readyToWriteJson
        with open(f"{frame_name}.json", "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        c= cv2.waitKey(1) & 0xff 

        if c==27:
            capture.release()
            break

def closeOpenCV():
    global ocv, folder_name, frames_info
    ocv = False 
    print("saved")
    fName = folder_name.split(".")[0]
    with open(f"{fName}/frames_info.json", "w") as json_file:
        json.dump(frames_info, json_file, indent=4)       

MainWindow.closeEvent = closeOpenCV  

def keyPressEvent(event):
    global autoPilot, lastPlay, folder_name, frames_info, controlAngle
    # event.key() 會返回被按下的按鍵的鍵碼
    commandNum = 0
    if event.key() == QtCore.Qt.Key_A:
        controlAngle = 40
        siteValue.setValue(0)
    elif event.key() == QtCore.Qt.Key_S:
        controlAngle = 70
        siteValue.setValue(1)
    elif event.key() == QtCore.Qt.Key_D:
        controlAngle = 90
        siteValue.setValue(2)
    elif event.key() == QtCore.Qt.Key_F:
        controlAngle = 110
        siteValue.setValue(3)
    elif event.key() == QtCore.Qt.Key_G:
        controlAngle = 140
        siteValue.setValue(4)
    elif event.key() == QtCore.Qt.Key_Escape:
        commandNum = 300
        autoPilot = False
        play_sound("sound/autoPilotOFF.wav")
    elif event.key() == QtCore.Qt.Key_Q:
        commandNum = 301
        autoPilot = True
        play_sound("sound/autoPilotON.wav")
    elif event.key() == QtCore.Qt.Key_U:
        commandNum = 411
    elif event.key() == QtCore.Qt.Key_I:
        commandNum = 410
    elif event.key() == QtCore.Qt.Key_O:
        commandNum = 421
    elif event.key() == QtCore.Qt.Key_P:
        commandNum = 420
    elif event.key() == QtCore.Qt.Key_T:
        playWarning()
    elif event.key() == QtCore.Qt.Key_W:
        commandNum = 501
    elif event.key() == QtCore.Qt.Key_E:
        commandNum = 500
    # elif event.key() == QtCore.Qt.Key_M:
    #     print("saved")
    #     print(frames_info)
    #     with open(f"{folder_name}/frames_info.json", "w") as json_file:
    #         json.dump(frames_info, json_file, indent=4)  
    #     time.sleep(4)
    #     QtWidgets.QCoreApplication.quit()

    if commandNum != 0:
        global ser, openSerial
        print("motor command sended")
        if openSerial:
            ser.write((str(commandNum)+"\n").encode())
            print("Command "+str(commandNum)+"sended\n")

video = threading.Thread(target=opencv)
video.start()

mapThread = threading.Thread(target=startMap)
mapThread.start()

MainWindow.keyPressEvent = keyPressEvent
MainWindow.show()
# TrapezoidWindow.show()

sys.exit(app.exec_())