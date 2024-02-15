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

import simpleaudio as sa
from scipy.signal import convolve

autoPilot = False

openSerial = False
cameraUse = False

if openSerial:
    print("Wait connect")
    COM_PORT = '/dev/cu.usbmodem1401'
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

mapLabel = QtWidgets.QLabel(MainWindow)
mapLabel.setGeometry(555, -10, 150, 150)

y = 435
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

def keyPressEvent(event):
    global autoPilot
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
    # data = {"middlePointX": data["middlePointX"], "middlePointY": data["middlePointY"]}
    with open("setting.json", 'w') as file:
        json.dump(data, file)
    print("saved")
shortcut1 = QtWidgets.QShortcut(QKeySequence("Ctrl+S"), MainWindow)
shortcut1.activated.connect(save)

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
def slidingWindow(frame):
    global data, colors, site, openSerial, lastTime, autoPilot, lastRoute
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
                            x1 = int((1919-rectWidth)/2)
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
                            biggest["dist"] = abs(blockPercent[0]-blockPercent[1])
                            biggest["cdn"] = block
                    else:
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
                            x1 = int((1919-rectWidth)/2)
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
            # testPoints_array = np.array(testPoints, dtype=np.int32)
            # cv2.polylines(frame, [testPoints_array], isClosed=False, color=(235, 0, 0), thickness=20)
            if autoPilot:
                cv2.rectangle(frame, (block[0][0], cdnY-rectHeight), (block[1][1], cdnY), rectColor, 4, cv2.LINE_AA)
        #     cv2.putText(frame, str(blockPercent[0]), (block[0][0]-130, cdnY-10), cv2.FONT_HERSHEY_SIMPLEX,
        # 2, (0, 255, 255), 4, cv2.LINE_AA)
        #     cv2.putText(frame, str(blockPercent[1]), (block[1][1]+10, cdnY-10), cv2.FONT_HERSHEY_SIMPLEX,
        # 2, (0, 255, 255), 4, cv2.LINE_AA)
        cdnY -= rectHeight
        rectWidth -= adjustNum
        rectWidth = max(rectWidth, 0)

    points_array = np.array(points, dtype=np.int32)
    if site == 0 or site == 4:
        points_array = points_array[:10]
    elif site == 2:
        points_array = points_array[:15]

    cv2.polylines(frame, [points_array], isClosed=False, color=(235, 99, 169), thickness=40)
    if blockPercent[0]+blockPercent[1] < 10:
        rectColor = (200, 0, 0)
    coords = points_array
    if site != 2:
        median_coords = np.median(coords, axis=0)
    elif site == 1 or site == 3:
        median_coords = np.mean(coords, axis=0)
    else:
        maxCdn = np.max(coords, axis=0)
        minCdn = np.min(coords, axis=0)
        if abs(maxCdn[0]-959) > abs(minCdn[0]-959):
            median_coords = maxCdn
        else:
            median_coords = minCdn
        
    cv2.circle(frame, (int(median_coords[0]), int(median_coords[1])), 15, (181, 99, 235), -1)
    if site >= 3:
        point_coords = np.array([1200, 1079])
    elif site <= 1:
        point_coords = np.array([718, 1079])
    else:
        point_coords = np.array([959, 1079])
    relative_coords = point_coords - median_coords
    angle_rad = np.arctan2(relative_coords[1], relative_coords[0])
    angle_deg = np.degrees(angle_rad)

    if site == 1 or site == 3:
        if angle_deg >= 120 or angle_deg <= 60:
            muiltNum = 1.2
        else:
            muiltNum = 0.5
    elif site == 2:
        muiltNum = 0.7
    elif site == 4:
        if angle_deg >= 130:
            muiltNum = 1.4
        elif angle_deg <= 80:
            muiltNum = 1.2
        else:
            muiltNum = 0.7
    elif site == 0:
        if angle_deg <= 50:
            muiltNum = 1.4
        elif angle_deg >= 100:
            muiltNum = 1.2
        else:
            muiltNum = 0.7
    angle_deg = int(max(min(90+(angle_deg-90)*muiltNum, 180), 0))

    fps = round(1.0/(time.time()-lastTime), 2)
    lastTime = time.time()
    fpsText.setText("Fps: "+str(fps))
    angleText.setText("Angle: "+str(angle_deg))
    # print("fps= %.2f, angle= %4d"%(fps, angle_deg), end='\r')
    if openSerial:
        global ser
        ser.write((str(int(angle_deg))+'\n').encode())

    return frame
class DeeplabV3(object):
    _defaults = {
        "model_path"        : 'model/3_7.h5',
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

# mapImg = output = np.zeros((150, 150, 3), dtype="uint8")
def getMap():

    global mapLabel
    # 取得座標資料
    api_url = "http://xhinherpro.xamjiang.com/getData"
    response = requests.get(api_url)
    data = response.json()
    print(data)
    latitude = data["latitude"]
    longitude = data["longitude"]
    angle = int(data["site"])

    if latitude > 50:
        temp = longitude
        longitude = latitude
        latitude = temp

    # 獲取地圖數據
    G = ox.graph_from_point((latitude, longitude), dist=400, network_type='drive_service')
    print(G)
    origin = ox.distance.nearest_nodes(G, longitude, latitude)
    destination = ox.distance.nearest_nodes(G, 121.32012, 24.99422)

    route = nx.shortest_path(G, origin, destination)
    # ox.plot_graph_route(G, route)

    # 取得所有點的座標
    nodes = list(G.nodes())
    node_coordinates = []

    for node in nodes:
        x, y = G.nodes[node]['x'], G.nodes[node]['y']
        node_coordinates.append((x, y))


    # 取得所有線條的座標
    edges = list(G.edges())
    edge_coordinates = []
    for edge in edges:
        u, v = edge
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        edge_coordinates.append([(x1, y1), (x2, y2)])
        
    # 計算最右最左最上最下的值
    min_x = min([x for x, _ in node_coordinates])
    min_y = min([y for _, y in node_coordinates])
    max_x = max([x for x, _ in node_coordinates])
    max_y = max([y for _, y in node_coordinates])

    # 等比例轉換成能放進去500*500的OpenCV空白畫面內顯示
    width = 150
    height = 150
    scale_x = width / (max_x - min_x)
    scale_y = height / (max_y - min_y)

    # 繪製地圖
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img.fill(255)
    for u, v in edge_coordinates:
        ux = int((u[0] - min_x) * scale_x)
        uy = int((u[1] - min_y) * scale_y)
        vx = int((v[0] - min_x) * scale_x)
        vy = int((v[1] - min_y) * scale_y)
        cv2.line(img, (ux, uy), (vx, vy), (248, 242, 241), 4)

    for x, y in node_coordinates:
        x = int((x - min_x) * scale_x - 1)
        y = int((y - min_y) * scale_y - 1)
        # img[y, x] = (255, 255, 255)
        cv2.circle(img, (x, y), 2, (126, 201, 255), 2, -1)

    cv2.circle(img, (int((longitude - min_x) * scale_x - 1), int((latitude - min_y) * scale_y - 1)), 3, (255, 205, 125), 3, -1)
    img = cv2.flip(img, 0)

    #     # 將 NumPy 陣列轉換為 QImage
    # qimage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)

    # # 創建 QPainter 對象
    # painter = QPainter()

    # # 將 QImage 繪製到畫布上
    # painter.drawImage(0, 0, qimage)

    # # 繪製圓形
    # painter.drawEllipse(QtCore.QPoint(150, 150), 150, 150)

    # 將畫布轉換為 QPixmap
    # mapImg = QImage(img, 250, 250, 250*3, QImage.Format_RGB888)
    # mapLabel.setPixmap(QPixmap.fromImage(mapImg))
    mapLabel.setPixmap(mask_image(img, angle)) 

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
    
    # getMap()

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