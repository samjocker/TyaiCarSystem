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

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import requests

openSerial = False
cameraUse = False

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

label = QtWidgets.QLabel(MainWindow)
label.setGeometry(0, 0, 720, 405)

y = 435
fpsText = QtWidgets.QLabel(MainWindow)
fpsText.setGeometry(70, y, 120, 30)
font = QFont() 
font.setPointSize(24)
fpsText.setFont(font)
fpsText.setText("Fps: "+str(0))
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
rectAdjustValue.setMaximum(100)
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

def save():
    global data
    # data = {"middlePointX": data["middlePointX"], "middlePointY": data["middlePointY"]}
    with open("setting.json", 'w') as file:
        json.dump(data, file)
    print("saved")
shortcut1 = QtWidgets.QShortcut(QKeySequence("Ctrl+S"), MainWindow)
shortcut1.activated.connect(save)

def map(value, from_low, from_high, to_low, to_high):
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low

lastTime = time.time()
def slidingWindow(frame):
    global data, colors, site, openSerial, lastTime
    rectColor = (0, 200 ,0)

    cdnY = 1079
    rectHeight = 50
    rectWidth = int(data["rectWidth"])
    adjustNum = int(rectWidth*(data["rectAdjust"]/100))
    suggestSite = True
    points = []
    # [959, 1079]
    # TODO:add turn site identify
    while (cdnY-rectHeight >= 0):
        x1 = int((1919-rectWidth)/2)
        if rectWidth > 0:
            block = [[x1,x1+int(rectWidth/2)], [x1+int(rectWidth/2), x1+rectWidth]]
            blockPercent = [0, 0]
            keepAdjust = True
            runTime = 0
            lastBlock = []
            addNum = 40
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

                if site == 0:
                    # addNum = -20

                    
                    if block[0][0] < 0:
                        if lastBlock != []:
                            keepAdjust = False
                            points.append([lastBlock[0][1], cdnY-rectHeight])
                            block = lastBlock
                            break
                        else:
                            addNum = -40
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
                    elif abs(blockPercent[0]-blockPercent[1]) <= 20 and blockPercent[0] > 5:
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
                        addNum = -40
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
                            addNum = -40
                        elif blockPercent[0] < blockPercent[1] and blockPercent[0] > 10:
                            suggestSite = True
                            addNum = 40
                        else:
                            if suggestSite:
                                addNum = 40
                            else:
                                addNum = -40
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
                        addNum = -40
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
                            x1 = int((1919-rectWidth)/2)
                            addNum = -40
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
                    elif abs(blockPercent[0]-blockPercent[1]) <= 20 and blockPercent[0] > 5:
                        lastBlock = block
                        if abs(blockPercent[0]-blockPercent[1]) < biggest["dist"]:
                            biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                            biggest["cdn"] = block
                    else:
                        if lastBlock != []:
                            keepAdjust = False
                            block = lastBlock
                            points.append([block[0][1], cdnY-rectHeight])
                            break
                        else:
                            if abs(blockPercent[0]-blockPercent[1]) < biggest["dist"] and blockPercent[0] > 5:
                                biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                                biggest["cdn"] = block
                    x1 += addNum
                # testPoints.append([block[0][1], cdnY-rectHeight])
                if runTime >= 100:
                    keepAdjust = False
                    # print(blockPercent)
                    print("break", blockPercent[0], blockPercent[1], site)

                    # print(f'顏色佔比: {percentage}%')
            # points.append([block[0][1], cdnY-rectHeight])
            testPoints_array = np.array(testPoints, dtype=np.int32)
            cv2.polylines(frame, [testPoints_array], isClosed=False, color=(235, 0, 0), thickness=20)
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
    #FIXME:site 4 turning strstegy
    if site == 1 or site == 2 or site == 3:
        muiltNum = 1.5 if angle_deg<110 and angle_deg>70 else 1.3
    elif site == 4:
        if angle_deg >= 90:
            muiltNum = 0.8 if angle_deg<110 else 0.7
        else:
            muiltNum = 1.5
    elif site == 0:
        if angle_deg <= 90:
            muiltNum = 0.8 if angle_deg>70 else 0.7
        else:
            muiltNum = 1.5
    angle_deg = max(min(90+(angle_deg-90)*muiltNum, 180), 0)

    fps = round(1.0/(time.time()-lastTime), 2)
    lastTime = time.time()
    fpsText.setText("Fps: "+str(fps))
    print("fps= %.2f, angle= %4d"%(fps, angle_deg), end='\r')
    if openSerial:
        global ser
        ser.write((str(int(angle_deg))+'\n').encode())

    return frame
class DeeplabV3(object):
    _defaults = {
        "model_path"        : 'model/3_5.h5',
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

            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')

            image = Image.fromarray(np.uint8(seg_img))

        return image

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
deeplab = DeeplabV3()

video_path      = "/Users/sam/Documents/MyProject/mixProject/TYAIcar/MLtraning/visualIdentityVideo/IMG_1413.MOV"
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
            if cameraUse:
                frame = cv2.flip(frame, 0)
                frame = cv2.flip(frame, 1)
            if not ref:
                ocv = False
                capture.release()
                break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        frame = np.array(deeplab.detect_image(frame))

        height, width, channel = 405, 720, 3
        frame = cv2.resize(frame, (width, height))
        bytesPerline = channel * width

        img = QImage(frame, width, height, bytesPerline, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img))
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