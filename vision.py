import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import colorsys
import copy
import time, sys, json
import threading
import serial

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *

from nets.deeplab import Deeplabv3
from utils.utils import cvtColor, preprocess_input, resize_image

openSerial = False
cameraUse = False

if openSerial:
    print("Wait connect")
    COM_PORT = '/dev/cu.usbmodem1101'
    BAUD_RATES = 9600
    ser = serial.Serial(COM_PORT, BAUD_RATES)
    print("Connect successfuly")
    time.sleep(2)
    print("start!")

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
MainWindow.setObjectName("MainWindow")
MainWindow.setWindowTitle("TYAI car")
MainWindow.resize(720, 595)

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
trapezoidYvalue.setMaximum(1080)
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
        datumXvalue.setText(str(data["middlePointX"]))
        datumYvalue.setText(str(data["middlePointY"]))
        sideDistValue.setText(str(data["sideDistValue"]))
        trapezoidXnum.setText(str(data["trapezoidXvalue"]))
        trapezoidYnum.setText(str(data["trapezoidYvalue"]))
except FileNotFoundError:
    data = {"middlePointX": 0, "middlePointY": 0, "sideDistValue":0, "trapezoidXvalue": 0, "trapezoidYvalue": 0}
    datumXvalue.setText(str(0))
    datumYvalue.setText(str(0))
    sideDistValue.setText(str(0))
    trapezoidXnum.setText(str(0))
    trapezoidYnum.setText(str(0))
    with open("setting.json", 'w') as file:
        json.dump(data, file)
except KeyError:
    data = {"middlePointX": 0, "middlePointY": 0, "sideDistValue":0, "trapezoidXvalue": 0, "trapezoidYvalue": 0}
    datumXvalue.setText(str(0))
    datumYvalue.setText(str(0))
    trapezoidXnum.setText(str(0))
    trapezoidYnum.setText(str(0))
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
        ser.write("300".encode())
        print("servo change \n")
shortcut2 = QtWidgets.QShortcut(QKeySequence("Ctrl+D"), MainWindow)
shortcut2.activated.connect(servoChange)

def map(value, from_low, from_high, to_low, to_high):
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low

def putInformation(frame):
    global data, edge, sideButtonState
    height, width, channel = frame.shape

    # srcPts = np.float32([[0,1079], [1919,1079], [data["trapezoidXvalue"], data["trapezoidYvalue"]], [1919-data["trapezoidXvalue"], data["trapezoidYvalue"]]])
    # dstPts = np.float32([[0, 1079], [1919, 1079], [0, 0], [1919, 0]])
    # perspective_matrix = cv2.getPerspectiveTransform(srcPts, dstPts)
    # testFrame = cv2.warpPerspective(frame, perspective_matrix, (1920, 1080))
    # testFrame = cv2.resize(testFrame, (720, 405))
    # img = QImage(testFrame, 720, 405, 720*3, QImage.Format_RGB888)
    # TestLabel.setPixmap(QPixmap.fromImage(img))

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

    # for point in contour_array:
    #     cv2.circle(frame, tuple(point), 3, (0, 0, 255), 5)

    # 抓取 x 和 y 座標
    x, y = contour_array[:, 0], contour_array[:, 1]

    # 使用 NumPy 的濾波函數進行平滑處理
    smoothed_x = np.convolve(x, np.ones(10)/10, mode='valid')
    smoothed_y = np.convolve(y, np.ones(10)/10, mode='valid')

    # 將平滑後的座標組合回去
    smoothed_contour = np.column_stack((smoothed_x, smoothed_y))

    # 在原始圖片上畫出平滑後的線條
    smoothed_contour = smoothed_contour.astype(int)
    cv2.polylines(frame, [smoothed_contour], isClosed=True, color=(123, 222, 245), thickness=8)

    # rectY = (int(data["middlePointY"]/45)+1)*45
    # frame = cv2.rectangle(frame, ((int(edge["offsetLeft"]/32)-1)*32, rectY), ((int(edge["offsetLeft"]/32)+1)*32, rectY-45), (0, 200, 0), 2, cv2.LINE_AA)

    frame = cv2.circle(frame, (data["middlePointX"], data["middlePointY"]), radius=5, color=(255,255,255), thickness=10)
    frame = cv2.circle(frame, (data["sideDistValue"], data["middlePointY"]), radius=5, color=(250,149,55), thickness=20)
    frame = cv2.line(frame, (0,1079), (data["trapezoidXvalue"], data["trapezoidYvalue"]), (4, 51, 96), 2)
    frame = cv2.line(frame, (1919,1079), (1919-data["trapezoidXvalue"], data["trapezoidYvalue"]), (4, 51, 96), 2)
    frame = cv2.line(frame, (data["trapezoidXvalue"], data["trapezoidYvalue"]), (1919-data["trapezoidXvalue"], data["trapezoidYvalue"]), (4, 51, 96), 2)
    frame = cv2.line(frame, (edge["offsetRight"], data["middlePointY"]), (edge["offsetLeft"], data["middlePointY"]), (255,255,255), 4)
    if data["middlePointY"] <= height/2:
        textOffset = 65
    else:
        textOffset = -20
    frame = cv2.putText(frame, str(edge["offsetLeft"]),(data["middlePointX"]-int(edge["offsetLeft"]/2)-60, data["middlePointY"]+textOffset), cv2.FONT_HERSHEY_SIMPLEX,
  2, (255, 255, 255), 8, cv2.LINE_AA)
    frame = cv2.putText(frame, str(edge["offsetRight"]),(data["middlePointX"]+int(edge["offsetRight"]/2)-60, data["middlePointY"]+textOffset), cv2.FONT_HERSHEY_SIMPLEX,
  2, (255, 255, 255), 8, cv2.LINE_AA)

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
        "model_path"        : 'model/3_2.h5',
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
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

            # kernel = np.ones((7,7),np.uint8)
            # seg_img = cv2.dilate(seg_img,kernel,iterations = 5)
            # seg_img = cv2.erode(seg_img,kernel,iterations = 5)
            getEdge(seg_img[data["middlePointY"]])
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

video_path      = "/Users/sam/Documents/MyProject/mixProject/TYAIcar/MLtraning/visualIdentityVideo/IMG_1309.MOV"
video_save_path = ""
video_fps       = 30.0

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
        for i in range(9):
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                ocv = False
                capture.release()
                break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        frame = np.array(deeplab.detect_image(frame))

        testFrame = perspective_correction(frame)
        testFrame = cv2.resize(testFrame, (720, 405))
        testImg = QImage(testFrame, 702, 405, 720*3, QImage.Format_RGB888)
        TestLabel.setPixmap(QPixmap.fromImage(testImg))

        frame = putInformation(frame)
        height, width, channel = 405, 720, 3
        frame = cv2.resize(frame, (width, height))
        bytesPerline = channel * width

        img = QImage(frame, width, height, bytesPerline, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img))

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        value = 0
        mapValue = [0, 0]

        if sideButtonState:
            value = edge["offsetRight"]-data["sideDistValue"]
            # mapValue = [int(data["middlePointX"]/2)*-1, int(data["middlePointX"]/2)]
            mapValue = [-960, 960]
            mapNum = max(min(map(value, mapValue[0], mapValue[1], 90, -90)+90, 180), 0)
        else:
            value = edge["offsetLeft"]-data["sideDistValue"]
            # mapValue = [int(data["middlePointX"]/2)*-1, int(data["middlePointX"]/2)]
            mapValue = [-960, 960]
            mapNum = max(min(map(value, mapValue[0], mapValue[1], -90, 90)+90, 180), 0)

        # print("fps= %.2f, angle= %4d"%(fps, mapNum), end='\r')
        if openSerial:
            global ser
            ser.write((str(int(mapNum))+'\n').encode())

        c= cv2.waitKey(1) & 0xff 
        if video_save_path!="":
            out.write(frame)

        if c==27:
            capture.release()
            break

video = threading.Thread(target=opencv)
video.start()

MainWindow.show()
# TrapezoidWindow.show()
sys.exit(app.exec_())