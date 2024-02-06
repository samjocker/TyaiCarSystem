import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import colorsys
import copy
import time, sys, json
import threading
import asyncio
import serial

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *

from nets.deeplab import Deeplabv3
from utils.utils import cvtColor, preprocess_input, resize_image

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import requests

openSerial = True
cameraUse = True

if openSerial:
    print("Wait connect")
    COM_PORT = '/dev/cu.usbmodem11301'
    BAUD_RATES = 9600
    ser = serial.Serial(COM_PORT, BAUD_RATES)
    print("Connect successfuly")
    time.sleep(2)
    print("start!")

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
MainWindow.setObjectName("MainWindow")
MainWindow.setWindowTitle("TYAI car")
MainWindow.resize(720, 685)

TrapezoidWindow = QtWidgets.QMainWindow()
TrapezoidWindow.setObjectName("TrapezoidWindow")
TrapezoidWindow.setWindowTitle("Test")
TrapezoidWindow.resize(720, 435)

TestLabel = QtWidgets.QLabel(TrapezoidWindow)    
TestLabel.setGeometry(0,0,720,405)


label = QtWidgets.QLabel(MainWindow)    
label.setGeometry(0,0,720,405)          
datumXslider = QtWidgets.QSlider(MainWindow)
datumXslider.setGeometry(110, 430, 500, 30)
datumXslider.setOrientation(QtCore.Qt.Horizontal)
datumXslider.setMaximum(1920)
datumXslider.setMinimum(-1920)
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

sideDistSlider = QtWidgets.QSlider(MainWindow)
sideDistSlider.setGeometry(110, 490, 380, 30)
sideDistSlider.setOrientation(QtCore.Qt.Horizontal)
sideDistSlider.setMaximum(1919)
sideDistTitle = QtWidgets.QLabel(MainWindow)
sideDistTitle.setGeometry(70, 487, 80, 30)
font = QFont() 
font.setPointSize(18)
sideDistTitle.setText("邊距")
sideDistTitle.setFont(font)
sideDistValue = QtWidgets.QLabel(MainWindow)
sideDistValue.setGeometry(500, 487, 80, 30)
font = QFont() 
font.setPointSize(18)
sideDistValue.setText("555")
sideDistValue.setFont(font)

sideButton = QtWidgets.QPushButton(MainWindow)
sideButton.setGeometry(535, 487, 80, 30)
sideButton.setText("Right")
sideButtonState = True

trapezoidXvalue = QtWidgets.QSlider(MainWindow)
trapezoidXvalue.setGeometry(110, 520, 500, 30)
trapezoidXvalue.setOrientation(QtCore.Qt.Horizontal)
trapezoidXvalue.setMaximum(10000)
trapezoidXvalue.setMinimum(0)
trapezoidXtitle = QtWidgets.QLabel(MainWindow)
trapezoidXtitle.setGeometry(70, 517, 80, 30)
font = QFont() 
font.setPointSize(18)
trapezoidXtitle.setText("梯X")
trapezoidXtitle.setFont(font)
trapezoidXnum = QtWidgets.QLabel(MainWindow)
trapezoidXnum.setGeometry(620, 517, 80, 30)
font = QFont() 
font.setPointSize(18)
trapezoidXnum.setText("555")
trapezoidXnum.setFont(font)

trapezoidYvalue = QtWidgets.QSlider(MainWindow)
trapezoidYvalue.setGeometry(110, 550, 500, 30)
trapezoidYvalue.setOrientation(QtCore.Qt.Horizontal)
trapezoidYvalue.setMaximum(1920)
trapezoidYtitle = QtWidgets.QLabel(MainWindow)
trapezoidYtitle.setGeometry(70, 547, 80, 30)
font = QFont() 
font.setPointSize(18)
trapezoidYtitle.setText("梯Y")
trapezoidYtitle.setFont(font)
trapezoidYnum = QtWidgets.QLabel(MainWindow)
trapezoidYnum.setGeometry(620, 547, 80, 30)
font = QFont() 
font.setPointSize(18)
trapezoidYnum.setText("555")
trapezoidYnum.setFont(font)

rectWidthValue = QtWidgets.QSlider(MainWindow)
rectWidthValue.setGeometry(110, 580, 500, 30)
rectWidthValue.setOrientation(QtCore.Qt.Horizontal)
rectWidthValue.setMaximum(1919)
rectWidthTitle = QtWidgets.QLabel(MainWindow)
rectWidthTitle.setGeometry(70, 577, 80, 30)
font = QFont() 
font.setPointSize(18)
rectWidthTitle.setText("方寬")
rectWidthTitle.setFont(font)
rectWidthNum = QtWidgets.QLabel(MainWindow)
rectWidthNum.setGeometry(620, 577, 80, 30)
font = QFont() 
font.setPointSize(18)
rectWidthNum.setText("555")
rectWidthNum.setFont(font)

rectAdjustValue = QtWidgets.QSlider(MainWindow)
rectAdjustValue.setGeometry(110, 610, 500, 30)
rectAdjustValue.setOrientation(QtCore.Qt.Horizontal)
rectAdjustValue.setMaximum(100)
rectAdjustTitle = QtWidgets.QLabel(MainWindow)
rectAdjustTitle.setGeometry(70, 607, 80, 30)
font = QFont() 
font.setPointSize(18)
rectAdjustTitle.setText("微調")
rectAdjustTitle.setFont(font)
rectAdjustNum = QtWidgets.QLabel(MainWindow)
rectAdjustNum.setGeometry(620, 607, 80, 30)
font = QFont() 
font.setPointSize(18)
rectAdjustNum.setText("555")
rectAdjustNum.setFont(font)

siteValue = QtWidgets.QSlider(MainWindow)
siteValue.setGeometry(110, 640, 500, 30)
siteValue.setOrientation(QtCore.Qt.Horizontal)
siteValue.setMaximum(4)
siteValue.setTickPosition(3)
siteValue.setValue(2)
# siteValue.setTickInterval(20)
siteTitle = QtWidgets.QLabel(MainWindow)
siteTitle.setGeometry(70, 637, 80, 30)
font = QFont() 
font.setPointSize(18)
siteTitle.setText("  左")
siteTitle.setFont(font)
siteNum = QtWidgets.QLabel(MainWindow)
siteNum.setGeometry(620, 637, 80, 30)
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
        datumXslider.setValue(data["middlePointX"])
        datumYslider.setValue(data["middlePointY"])
        sideDistSlider.setValue(data["sideDistValue"])
        trapezoidXvalue.setValue(data["trapezoidXvalue"])
        trapezoidYvalue.setValue(data["trapezoidYvalue"])
        rectWidthValue.setValue(data["rectWidth"])
        rectAdjustValue.setValue(data["rectAdjust"])
        datumXvalue.setText(str(data["middlePointX"]))
        datumYvalue.setText(str(data["middlePointY"]))
        sideDistValue.setText(str(data["sideDistValue"]))
        trapezoidXnum.setText(str(data["trapezoidXvalue"]))
        trapezoidYnum.setText(str(data["trapezoidYvalue"]))
        rectWidthNum.setText(str(data["rectWidth"]))
        rectAdjustNum.setText(str(data["rectAdjust"]))
except FileNotFoundError:
    data = {"middlePointX": 0, "middlePointY": 0, "sideDistValue":0, "trapezoidXvalue": 0, "trapezoidYvalue": 0, "rectWidth": 0, "rectAdjust": 0}
    datumXvalue.setText(str(0))
    datumYvalue.setText(str(0))
    sideDistValue.setText(str(0))
    trapezoidXnum.setText(str(0))
    trapezoidYnum.setText(str(0))
    rectWidthNum.setText(str(0))
    rectAdjustNum.setText(str(0))
    with open("setting.json", 'w') as file:
        json.dump(data, file)
except KeyError:
    data = {"middlePointX": 0, "middlePointY": 0, "sideDistValue":0, "trapezoidXvalue": 0, "trapezoidYvalue": 0, "rectWidth": 0, "rectAdjust": 0}
    datumXvalue.setText(str(0))
    datumYvalue.setText(str(0))
    sideDistValue.setText(str(0))
    trapezoidXnum.setText(str(0))
    trapezoidYnum.setText(str(0))
    rectWidthNum.setText(str(0))
    rectAdjustNum.setText(str(0))
    with open("setting.json", 'w') as file:
        json.dump(data, file)
except Exception as e:
    print(e)

def keyPressEvent(event):
    # event.key() 會返回被按下的按鍵的鍵碼
    commandNum = 0
    if event.key() == QtCore.Qt.Key_A:
        siteValue.setValue(0)
    elif event.key() == QtCore.Qt.Key_S:
        siteValue.setValue(1)
    elif event.key() == QtCore.Qt.Key_D:
        siteValue.setValue(2)
    elif event.key() == QtCore.Qt.Key_F:
        siteValue.setValue(3)
    elif event.key() == QtCore.Qt.Key_G:
        siteValue.setValue(4)
    elif event.key() == QtCore.Qt.Key_Escape:
        commandNum = 300
    elif event.key() == QtCore.Qt.Key_U:
        commandNum = 411
    elif event.key() == QtCore.Qt.Key_I:
        commandNum = 410
    elif event.key() == QtCore.Qt.Key_O:
        commandNum = 421
    elif event.key() == QtCore.Qt.Key_P:
        commandNum = 420

    if commandNum != 0:
        global ser, openSerial
        print("motor command sended")
        if openSerial:
            ser.write((str(commandNum)+"\n").encode())
            print("Command "+str(commandNum)+"sended\n")

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
def sideDistSliderValue(value):
    global data
    data["sideDistValue"] = value
    sideDistValue.setText(str(data["sideDistValue"]))
sideDistSlider.valueChanged.connect(sideDistSliderValue)
def sideButtonClicked():
    global sideButtonState
    if sideButtonState:
        sideButton.setText("Left")
    else:
        sideButton.setText("Right")
    sideButtonState = not sideButtonState
sideButton.clicked.connect(sideButtonClicked)
def trapezoidXvalueChange(value):
    data["trapezoidXvalue"] = value
    trapezoidXnum.setText(str(value))
trapezoidXvalue.valueChanged.connect(trapezoidXvalueChange)
def trapezoidYvalueChange(value):
    data["trapezoidYvalue"] = value
    trapezoidYnum.setText(str(value))
trapezoidYvalue.valueChanged.connect(trapezoidYvalueChange)
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
def closeOpenCV():
    global ocv
    ocv = False        

MainWindow.closeEvent = closeOpenCV  

def getEdge(pr):
    global data, colors
    leftOffset = 0
    rightOffset = 0
    
    # if np.all(pr[data["middlePointX"]-x] == list(colors[1])):

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
    #print("最長範圍：", highRange,end='\r')


    edge["offsetLeft"] = highRange[0]
    edge["offsetRight"] = highRange[1]

def save():
    global data
    # data = {"middlePointX": data["middlePointX"], "middlePointY": data["middlePointY"]}
    with open("setting.json", 'w') as file:
        json.dump(data, file)
    print("saved")
shortcut1 = QtWidgets.QShortcut(QKeySequence("Ctrl+S"), MainWindow)
shortcut1.activated.connect(save)

def servoChange():
    global ser, openSerial
    if openSerial:
        ser.write("300\n".encode())
        print("servo change \n")
shortcut2 = QtWidgets.QShortcut(QKeySequence("Ctrl+D"), MainWindow)
shortcut2.activated.connect(servoChange)

def map(value, from_low, from_high, to_low, to_high):
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low

def findLine(frame):
    # 定義特定顏色 (以BGR格式為例)
    color = np.array([255, 255, 255])

    # 建立遮罩，將特定顏色以外的像素設為黑色，其他設為白色
    mask = cv2.inRange(frame, color, color)

    # 侵蝕操作，可調整 kernel 的大小
    kernel_erode = np.ones((15, 15), np.uint8)
    eroded_mask = cv2.erode(mask, kernel_erode, iterations=4)

    # 膨脹操作，可調整 kernel 的大小
    kernel_dilate = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=4)

    # 尋找輪廓
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 將輪廓座標轉為numpy array
    contour_array = np.vstack(contours).squeeze()

    cv2.polylines(frame, [contour_array], isClosed=True, color=(123, 222, 245), thickness=20)
    return frame

def putInformation(frame):
    global data, edge, sideButtonState, colors
    height, width, channel = frame.shape

    # srcPts = np.float32([[0,1079], [1919,1079], [data["trapezoidXvalue"], data["trapezoidYvalue"]], [1919-data["trapezoidXvalue"], data["trapezoidYvalue"]]])
    # dstPts = np.float32([[0, 1079], [1919, 1079], [0, 0], [1919, 0]])
    # perspective_matrix = cv2.getPerspectiveTransform(srcPts, dstPts)
    # testFrame = cv2.warpPerspective(frame, perspective_matrix, (1920, 1080))
    # testFrame = cv2.resize(testFrame, (720, 405))
    # img = QImage(testFrame, 720, 405, 720*3, QImage.Format_RGB888)
    # TestLabel.setPixmap(QPixmap.fromImage(img))

    # rectY = (int(data["middlePointY"]/45)+1)*45
    # frame = cv2.rectangle(frame, ((int(edge["offsetLeft"]/32)-1)*32, rectY), ((int(edge["offsetLeft"]/32)+1)*32, rectY-45), (0, 200, 0), 2, cv2.LINE_AA)

    # frame = cv2.circle(frame, (data["middlePointX"], data["middlePointY"]), radius=5, color=(255,255,255), thickness=10)
    # frame = cv2.circle(frame, (data["sideDistValue"], data["middlePointY"]), radius=5, color=(250,149,55), thickness=20)
    frame = cv2.line(frame, (0,1079), (data["trapezoidXvalue"], data["trapezoidYvalue"]), (4, 51, 96), 2)
    frame = cv2.line(frame, (1919,1079), (1919-data["trapezoidXvalue"], data["trapezoidYvalue"]), (4, 51, 96), 2)
    frame = cv2.line(frame, (data["trapezoidXvalue"], data["trapezoidYvalue"]), (1919-data["trapezoidXvalue"], data["trapezoidYvalue"]), (4, 51, 96), 2)
    # frame = cv2.line(frame, (edge["offsetRight"], data["middlePointY"]), (edge["offsetLeft"], data["middlePointY"]), (255,255,255), 4)
#     if data["middlePointY"] <= height/2:
#         textOffset = 65
#     else:
#         textOffset = -20
#     frame = cv2.putText(frame, str(edge["offsetLeft"]),(data["middlePointX"]-int(edge["offsetLeft"]/2)-60, data["middlePointY"]+textOffset), cv2.FONT_HERSHEY_SIMPLEX,
#   2, (255, 255, 255), 8, cv2.LINE_AA)
#     frame = cv2.putText(frame, str(edge["offsetRight"]),(data["middlePointX"]+int(edge["offsetRight"]/2)-60, data["middlePointY"]+textOffset), cv2.FONT_HERSHEY_SIMPLEX,
#   2, (255, 255, 255), 8, cv2.LINE_AA)
        # rectHeight = int(rectHeight*(data["rectAdjust"]/100))

    return frame

def slidingWindow(frame):
    global data, colors, site, openSerial
    rectColor = (0, 200 ,0)

    cdnY = 1079
    rectHeight = 50
    rectWidth = int(data["rectWidth"])
    adjustNum = int(rectWidth*(data["rectAdjust"]/100))
    suggestSite = True
    points = []
    # [959, 1079]
    while (cdnY-rectHeight >= 0):
        x1 = int((1919-rectWidth)/2)
        x2 = x1 + rectWidth
        if rectWidth > 0:
            block = [[x1,x1+int(rectWidth/2)], [x1+int(rectWidth/2), x1+rectWidth]]
            blockPercent = [0, 0]
            keepAdjust = True
            runTime = 0
            lastBlock = []
            addNum = 20
            biggest = {"cdn": [], "dist": 50}
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

                if site == 0:
                    # addNum = -20

                    
                    if block[0][0] < 0:
                        if lastBlock != []:
                            keepAdjust = False
                            points.append([lastBlock[0][1], cdnY-rectHeight])
                            block = lastBlock
                            break
                        else:
                            addNum = -20
                    elif block[1][1] > 1919:
                        print("\nout\n")
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
                    elif abs(blockPercent[0]-blockPercent[1]) <= 10 and blockPercent[0] > 20:
                        lastBlock = block
                        if abs(blockPercent[0]-blockPercent[1]) < biggest["dist"]:
                            biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                            biggest["cdn"] = block
                    elif abs(blockPercent[0]-blockPercent[1]) > 10:
                        if lastBlock != []:
                            keepAdjust = False
                            block = lastBlock
                            points.append([block[0][1], cdnY-rectHeight])
                            break
                        else:
                            if abs(blockPercent[0]-blockPercent[1]) < biggest["dist"]:
                                biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                                biggest["cdn"] = block
                    x1 -= addNum

                elif site == 1:

                    if block[0][0] < 0:
                        # if biggest["cdn"] != []:
                        #     points.append(biggest["cdn"])
                        # keepAdjust = False
                        # break
                        if abs(blockPercent[0]-blockPercent[1]) <= 30 and blockPercent[0] > 5:
                            points.append([block[0][1], cdnY-rectHeight])
                            keepAdjust = False
                            break
                        addNum = -20
                    elif block[1][1] > 1919:
                        keepAdjust = False
                        break
                    elif abs(blockPercent[0]-blockPercent[1]) <= 30 and blockPercent[0] > 5:
                        # if biggest["cdn"] != []:
                        #     points.append(biggest["cdn"])
                        # else:
                        points.append([block[0][1], cdnY-rectHeight])
                        keepAdjust = False
                        break
                    else:
                        if abs(blockPercent[0]-blockPercent[1]) < biggest["dist"]:
                            biggest["cdn"] = [block[0][1], cdnY-rectHeight]
                            biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                        
                        x1 -= addNum

                elif site == 2:
                    if abs(blockPercent[0]-blockPercent[1]) <= 5:
                        if blockPercent[0] != 0:
                            points.append([block[0][1], cdnY-rectHeight])
                        keepAdjust = False
                    else:
                        addNum = 20
                        if blockPercent[0] > blockPercent[1]:
                            suggestSite = False
                            addNum = -20
                        elif blockPercent[0] < blockPercent[1]:
                            suggestSite = True
                            addNum = 20
                        elif blockPercent[0] == blockPercent[1]:
                            if not suggestSite:
                                addNum = -20
                            else:
                                addNum = 20
                        x1 += addNum

                elif site == 3:

                    if block[1][1] > 1919:
                        # if biggest["cdn"] != []:
                        #     points.append(biggest["cdn"])
                        # keepAdjust = False
                        # break
                        if abs(blockPercent[0]-blockPercent[1]) <= 30 and blockPercent[0] > 5:
                            points.append([block[0][1], cdnY-rectHeight])
                            keepAdjust = False
                            break
                        addNum = -20
                    elif block[0][0] < 0:
                        keepAdjust = False
                        break
                    elif abs(blockPercent[0]-blockPercent[1]) <= 30 and blockPercent[0] > 5:
                        # if biggest["cdn"] != []:
                        #     points.append(biggest["cdn"])
                        # else:
                        points.append([block[0][1], cdnY-rectHeight])
                        keepAdjust = False
                        break
                    else:
                        if abs(blockPercent[0]-blockPercent[1]) < biggest["dist"]:
                            biggest["cdn"] = [block[0][1], cdnY-rectHeight]
                            biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                        
                        x1 += addNum

                elif site == 4:
                    # addNum = 20

                    if block[1][1] > 1919:
                        if lastBlock != []:
                            keepAdjust = False
                            points.append([lastBlock[0][1], cdnY-rectHeight])
                            block = lastBlock
                            break
                        else:
                            addNum = -20
                    elif block[0][0] < 0:
                        print("\nout\n")
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
                    elif abs(blockPercent[0]-blockPercent[1]) <= 10 and blockPercent[0] > 20:
                        lastBlock = block
                        if abs(blockPercent[0]-blockPercent[1]) < biggest["dist"]:
                            biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                            biggest["cdn"] = block
                    elif abs(blockPercent[0]-blockPercent[1]) > 10:
                        if lastBlock != []:
                            keepAdjust = False
                            block = lastBlock
                            points.append([block[0][1], cdnY-rectHeight])
                            break
                        else:
                            if abs(blockPercent[0]-blockPercent[1]) < biggest["dist"]:
                                biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                                biggest["cdn"] = block
                    x1 += addNum

                if runTime >= 100:
                    keepAdjust = False
                    # print(blockPercent)
                    print("break", block[1][1], site)

                    # print(f'顏色佔比: {percentage}%')
            # points.append([block[0][1], cdnY-rectHeight])
                    
            cv2.rectangle(frame, (block[0][0], cdnY-rectHeight), (block[1][1], cdnY), rectColor, 4, cv2.LINE_AA)
        #     cv2.putText(frame, str(blockPercent[0]), (block[0][0]-130, cdnY-10), cv2.FONT_HERSHEY_SIMPLEX,
        # 2, (0, 255, 255), 4, cv2.LINE_AA)
        #     cv2.putText(frame, str(blockPercent[1]), (block[1][1]+10, cdnY-10), cv2.FONT_HERSHEY_SIMPLEX,
        # 2, (0, 255, 255), 4, cv2.LINE_AA)
        cdnY -= rectHeight
        rectWidth -= adjustNum
        rectWidth = max(rectWidth, 0)

    points_array = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [points_array], isClosed=False, color=(235, 99, 169), thickness=40)
    if blockPercent[0]+blockPercent[1] < 10:
        rectColor = (200, 0, 0)
    coords = np.array(points)
    median_coords = np.median(coords, axis=0)
    cv2.circle(frame, (int(median_coords[0]), int(median_coords[1])), 15, (181, 99, 235), -1)
    if site == 4:
        point_coords = np.array([1050, 1079])
    elif site == 0:
        point_coords = np.array([880, 1079])
    else:
        point_coords = np.array([959, 1079])
    relative_coords = point_coords - median_coords
    angle_rad = np.arctan2(relative_coords[1], relative_coords[0])
    angle_deg = np.degrees(angle_rad)
    if site == 1 or site == 2 or site == 3:
        muiltNum = 1.5 if angle_deg<110 and angle_deg>70 else 1.3
    else:
        muiltNum = 0.8 if angle_deg<110 and angle_deg>70 else 0.7
    angle_deg = max(min(90+(angle_deg-90)*muiltNum, 180), 0)
    print("fps= %.2f, angle= %4d"%(6, angle_deg), end='\r')
    if openSerial:
        global ser
        ser.write((str(int(angle_deg))+'\n').encode())

    return frame

def perspective_correction(image):
    global data

    # image = cv2.resize(image, (864, 480))

    # 定義原始四邊形的四個點
    original_points = np.float32([[data["middlePointX"], data["trapezoidYvalue"]], [1920-data["middlePointX"], data["trapezoidYvalue"]], [data["trapezoidXvalue"]*-1, 1079], [data["trapezoidXvalue"]+1920, 1079]])

    # 定義梯形校正後的四個點
    corrected_points = np.float32([[0, 0], [1919, 0], [0, 1079], [1919, 1079]])

    #  # 定義原始四邊形的四個點
    # original_points = np.float32([[0, 500], [0, 1420], [1079, 0], [1079, 1919]])

    # # 定義梯形校正後的四個點
    # corrected_points = np.float32([[0, 0], [0, 1919], [1079, 0], [1079, 1919]])

    # 計算透視變換矩陣
    perspective_matrix = cv2.getPerspectiveTransform(original_points, corrected_points)

    # 執行透視變換
    result = cv2.warpPerspective(image, perspective_matrix, (1920, 1080))

    return result

class DeeplabV3(object):
    _defaults = {
        "model_path"        : 'model/3_3.h5',
        "num_classes"       : 7,
        "backbone"          : "mobilenet",
        "input_shape"       : [387, 688],
        "downsample_factor" : 16,
        "mix_type"          : 0,
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
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

            # kernel = np.ones((7,7),np.uint8)
            # seg_img = cv2.dilate(seg_img,kernel,iterations = 5)
            # seg_img = cv2.erode(seg_img,kernel,iterations = 5)
            slidingWindow(seg_img)
            getEdge(seg_img[data["middlePointY"]])
            # findLine(seg_img)
            image   = Image.fromarray(np.uint8(seg_img))

            image   = Image.blend(old_img, image, 0.5)

        elif self.mix_type == 1:
            
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

            image   = Image.fromarray(np.uint8(seg_img))
            getEdge(seg_img[data["middlePointY"]])

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')

            image = Image.fromarray(np.uint8(seg_img))
            getEdge(seg_img[data["middlePointY"]])

        return image

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
deeplab = DeeplabV3()

video_path      = "/Users/sam/Documents/MyProject/mixProject/TYAIcar/MLtraning/visualIdentityVideo/IMG_1460.MOV"
video_save_path = ""
video_fps       = 30.0

mapImg = output = np.zeros((400, 400, 3), dtype="uint8")
def getMap():
    # 取得座標資料
    api_url = "http://xhinherpro.xamjiang.com/getData"
    response = requests.get(api_url)
    data = response.json()
    print(data)
    latitude = data["latitude"]
    longitude = data["longitude"]

    # 獲取地圖數據
    G = ox.graph_from_point((longitude, latitude), dist=150, network_type='drive_service')

    # 繪製地圖並設定路徑和背景的顏色
    fig, ax = ox.plot_graph(G, show=False, close=False, figsize=(10, 10), edge_color='white', bgcolor='gray', edge_linewidth=10.0)

    # 將Matplotlib圖像轉換為OpenCV格式
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())

    # 對圖像進行顏色和對比度調整（可以根據需要進行更進一步的調整）
    alpha = 1.5  # 控制對比度
    beta = 30    # 控制亮度
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    foreground = cv2.resize(adjusted_img, (400, 400))
    output = np.zeros((400, 400, 3), dtype="uint8")
    center_coordinates = (200, 200)
    radius = 200
    color = (255, 255, 255)  # 白色

    cv2.circle(output, center_coordinates, radius, color, -1)

    mapImg = cv2.bitwise_and(foreground, output)


def opencv():
    global ocv,video_path,video_save_path,video_fps, sideButtonState, openSerial
    
    if not cameraUse:
        capture=cv2.VideoCapture(video_path)
    else:
        capture=cv2.VideoCapture(0)
    if video_save_path!="":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    ref, frame = capture.read()
    if not ref:
        raise ValueError("Video source Error")

    fps = 0.0
    while(ocv):
        for i in range(1 if cameraUse else 9):
            t1 = time.time()
            ref, frame = capture.read()
            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)
            if not ref:
                ocv = False
                capture.release()
                break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        frame = np.array(deeplab.detect_image(frame))

        # tryFunc = threading.Thread(target=testFunc)
        # tryFunc.start()

        # testFrame = perspective_correction(frame)
        # testFrame = cv2.resize(testFrame, (720, 405))
        # testImg = QImage(testFrame, 702, 405, 720*3, QImage.Format_RGB888)
        # TestLabel.setPixmap(QPixmap.fromImage(testImg))

        frame = putInformation(frame)
        height, width, channel = 405, 720, 3
        frame = cv2.resize(frame, (width, height))
        bytesPerline = channel * width

        img = QImage(frame, width, height, bytesPerline, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img))
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        value = 0
        mapValue = [0, 0]
        # print("fps= %.2f, angle= %4d"%(fps, 90), end='\r')

        c= cv2.waitKey(1) & 0xff 
        if video_save_path!="":
            out.write(frame)

        if c==27:
            capture.release()
            break

video = threading.Thread(target=opencv)
video.start()

MainWindow.keyPressEvent = keyPressEvent
MainWindow.show()
# TrapezoidWindow.show()
sys.exit(app.exec_())