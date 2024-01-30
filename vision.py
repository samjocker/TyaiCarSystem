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

openSerial = True
cameraUse = True

if openSerial:
    print("Wait connect")
    COM_PORT = '/dev/cu.usbmodem13101'
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
sideDistSlider.setMaximum(960)
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
trapezoidXvalue.setMaximum(960)
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
trapezoidYvalue.setMaximum(540)
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

colors = [ (255, 255, 255), (150, 150, 150), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
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
    for x in range(data["middlePointX"]):
        if np.all(pr[data["middlePointX"]-x] == list(colors[1])):
            leftOffset += 1
        else:
            break
    prLong = len(pr)
    for x in range(prLong-data["middlePointX"]):
        if np.all(pr[x+data["middlePointX"]-1] == list(colors[1])):
            rightOffset += 1
        else:
            break
    edge["offsetLeft"] = leftOffset
    edge["offsetRight"] = rightOffset

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

    srcPts = np.float32([[0,1079], [1919,1079], [data["trapezoidXvalue"], data["trapezoidYvalue"]], [1919-data["trapezoidXvalue"], data["trapezoidYvalue"]]])
    dstPts = np.float32([[0, 1079], [1919, 1079], [0, 0], [1919, 0]])
    perspective_matrix = cv2.getPerspectiveTransform(srcPts, dstPts)
    testFrame = cv2.warpPerspective(frame, perspective_matrix, (1920, 1080))
    testFrame = cv2.resize(testFrame, (720, 405))
    img = QImage(testFrame, 720, 405, 720*3, QImage.Format_RGB888)
    TestLabel.setPixmap(QPixmap.fromImage(img))

    frame = cv2.circle(frame, (data["middlePointX"], data["middlePointY"]), radius=5, color=(255,255,255), thickness=10)
    sideCdn = ()
    if sideButtonState:
        sideCdn = (data["sideDistValue"]+data["middlePointX"], data["middlePointY"])
    else:
        sideCdn = (data["middlePointX"]-data["sideDistValue"], data["middlePointY"])
    frame = cv2.circle(frame, sideCdn, radius=5, color=(250,149,55), thickness=20)
    frame = cv2.line(frame, (0,1079), (data["trapezoidXvalue"], data["trapezoidYvalue"]), (255,255,255), 2)
    frame = cv2.line(frame, (1919,1079), (1919-data["trapezoidXvalue"], data["trapezoidYvalue"]), (255,255,255), 2)
    frame = cv2.line(frame, (data["trapezoidXvalue"], data["trapezoidYvalue"]), (1919-data["trapezoidXvalue"], data["trapezoidYvalue"]), (255,255,255), 2)
    frame = cv2.line(frame, (data["middlePointX"], data["middlePointY"]), (data["middlePointX"]-edge["offsetLeft"], data["middlePointY"]), (255,255,255), 4)
    frame = cv2.line(frame, (data["middlePointX"], data["middlePointY"]), (data["middlePointX"]+edge["offsetRight"], data["middlePointY"]), (255,255,255), 4)
    if data["middlePointY"] <= height/2:
        textOffset = 65
    else:
        textOffset = -20
    frame = cv2.putText(frame, str(edge["offsetLeft"]),(data["middlePointX"]-int(edge["offsetLeft"]/2)-60, data["middlePointY"]+textOffset), cv2.FONT_HERSHEY_SIMPLEX,
  2, (255, 255, 255), 8, cv2.LINE_AA)
    frame = cv2.putText(frame, str(edge["offsetRight"]),(data["middlePointX"]+int(edge["offsetRight"]/2)-60, data["middlePointY"]+textOffset), cv2.FONT_HERSHEY_SIMPLEX,
  2, (255, 255, 255), 8, cv2.LINE_AA)

    return frame

class DeeplabV3(object):
    _defaults = {
        "model_path"        : 'model/ep100-loss0.153-val_loss0.047.h5',
        "num_classes"       : 7,
        "backbone"          : "mobilenet",
        "input_shape"       : [512, 512],
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

            image   = Image.blend(old_img, image, 0.7)

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

video_path      = "/Users/sam/Documents/MyProject/mixProject/TYAIcar/MLtraning/visualIdentityVideo/IMG_9590.MOV"
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
        t1 = time.time()
        ref, frame = capture.read()
        if not ref:
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        frame = np.array(deeplab.detect_image(frame))
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
            mapValue = [int(1919-data["middlePointX"]-data["sideDistValue"])*-1, int(data["sideDistValue"])]
            mapNum = max(min(map(value, mapValue[0], mapValue[1], -90, 90)+90, 180), 0)
        else:
            value = edge["offsetLeft"]-data["sideDistValue"]
            # mapValue = [int(data["middlePointX"]/2)*-1, int(data["middlePointX"]/2)]
            mapValue = [int(data["middlePointX"]-data["sideDistValue"])*-1, int(data["sideDistValue"])]
            mapNum = max(min(map(value, mapValue[0], mapValue[1], 90, -90)+90, 180), 0)
        print("fps= %.2f, angle= %4d"%(fps, mapNum), end='\r')
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