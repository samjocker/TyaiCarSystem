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
from PIL import Image, ImageQt
from PyQt5.QtGui import QPixmap

from PyQt5.QtGui import QImage, QPixmap
from PIL import Image

from nets.deeplab import Deeplabv3
from utils.utils import cvtColor, preprocess_input, resize_image



import math, os
import serial

import requests

import threading

import datetime



shared_gps_data = {"latitude": 0, "longitude": 0, "site": "", "loraState": ""}
gps_data_lock = threading.Lock()



new_log = {
            "log_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "開始": "開始紀錄",
}

# # 讀取現有的 JSON 檔案
# with open("gpsLog.json", "r", encoding="utf-8") as json_file:
#     existing_data = json.load(json_file)

#  # 將新的 log 資訊加入到現有的資料中
# existing_data["logs"].append(new_log)

# # 將更新後的資料寫回 JSON 檔案
# with open("gpsLog.json", "w", encoding="utf-8") as json_file:
#     json.dump(existing_data, json_file, ensure_ascii=False, indent=2)




def update_gps_data():
    global shared_gps_data
    while True:
        # GPS
        url = "https://3908-60-251-221-219.ngrok-free.app/getData"
        try:
            response = requests.get(url)

            if response.status_code == 200:
                GPSdata = response.json()
                latitude = GPSdata['latitude']
                longitude = GPSdata['longitude']
                site = GPSdata['site']
                loraState = GPSdata['loraState']
            else:
                latitude = 0
                longitude = 0
                site = "None"
                loraState = "None"

            # 使用互斥鎖保護對 shared_gps_data 的訪問
            with gps_data_lock:
                shared_gps_data["latitude"] = latitude
                shared_gps_data["longitude"] = longitude
                shared_gps_data["site"] = site 
                shared_gps_data["loraState"] = loraState
            
            new_log = {
                "log_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "latitude": latitude,
                "longitude": longitude,
                "site": site,
                "loraState": loraState
            }

            # # 讀取現有的 JSON 檔案
            # with open("gpsLog.json", "r", encoding="utf-8") as json_file:
            #     existing_data = json.load(json_file)

            # # 將新的 log 資訊加入到現有的資料中
            # existing_data["logs"].append(new_log)

            # # 將更新後的資料寫回 JSON 檔案
            # with open("gpsLog.json", "w", encoding="utf-8") as json_file:
            #     json.dump(existing_data, json_file, ensure_ascii=False, indent=2)

            # 延遲一段時間，以免過於頻繁更新
            time.sleep(0.5)
        
        except:
            pass



openSerial = True

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
    COM_PORT = '/dev/cu.usbmodem1301'
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

colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
          (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
          (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
          (128, 64, 12)]


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

#video_path = r"D:\Data\project\tyaiCar\TyaiCarSystem\IMG_1461.MOV"
video_path = r"/Volumes/YihuanMiSSD/IMG_1460.MOV"
#video_path = r"D:/IMG_1319.MOV"

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

def getEdge(pr):

    #print(len(pr))

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
    #print(f"最長範圍：{highRange}         ",end='\r')

    return highRange

def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min


#  opencv set

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
datumYslider.setMaximum(2)
datumYslider.setMinimum(0)
datumYslider.setValue(1)

vdatumYslider = QtWidgets.QSlider(MainWindow)
vdatumYslider.setGeometry(0,520, 864, 30)
vdatumYslider.setOrientation(QtCore.Qt.Horizontal)
vdatumYslider.setMaximum(190)
vdatumYslider.setMinimum(-180)
vdatumYslider.setValue(190)

autoPilot = False

AdatumYslider = QtWidgets.QSlider(MainWindow)
AdatumYslider.setGeometry(0,540, 864, 30)
AdatumYslider.setOrientation(QtCore.Qt.Horizontal)
AdatumYslider.setMaximum(0)
AdatumYslider.setMinimum(1)
AdatumYslider.setValue(0)

servoState = False

def keyPressEvent(event):
    global servoState
    servoState = not servoState
    if event.key() == QtCore.Qt.Key_Escape:
        global ser, openSerial
        print("motor command sended")
        if openSerial:
            ser.write("300\n".encode())
            print("servo change \n")

ocv = True            
def closeOpenCV():
    global ocv
    ocv = False        

MainWindow.closeEvent = closeOpenCV  


# trapezoid_label = QtWidgets.QLabel(MainWindow)
# trapezoid_label.setGeometry(550, 0, 250, 160)
# trapezoid_label.setStyleSheet("QLabel { background-color : white; color : black; }")
# trapezoid_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)


# box

# line Box

lineBoxPoint = []
middlePointX = 432
boxWidth = 20
for i in range(240, 420, 20):
    lineBoxPoint.append([[middlePointX-boxWidth,i],[middlePointX+boxWidth,i-20]])
    boxWidth += 20

# turn left box

turnLeftBoxPoint = []
middlePointX = 350
boxWidth = 10
for i in range(289, 400, 20):
    turnLeftBoxPoint.append([[middlePointX-boxWidth,i],[middlePointX+boxWidth,i-20]])
    #boxWidth += 20

# turn right box
    
turnRightBoxPoint = []
middlePointX = 514
boxWidth = 10
for i in range(289, 400, 20):
    turnRightBoxPoint.append([[middlePointX-boxWidth,i],[middlePointX+boxWidth,i-20]])
    #boxWidth += 20

# run
useCam = True
CamID = 0


def opencv():
    global ocv

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

    fps = 0.0

    videoSpeed = 10
    if useCam:
        videoSpeed = 1



    gps_thread = threading.Thread(target=update_gps_data)
    gps_thread.start()


    

    while ocv:


        t1 = time.time()

        for i in range(videoSpeed):
            ref, frame = capture.read()

        frame =  cv2.resize(frame, (864, 480))
        
        if useCam:
            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)



        if not ref:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))

        deeplab.mix_type = 0
        result_img_blend, rd,result_img_trapezoid2 , modelOutput = deeplab.detect_image(frame)

        #print(modelOutput[400])
        height, width, channel = 480, 864, 3
        frame_blend = cv2.resize(np.array(result_img_blend), (width, height))

        offsetListLeft = []
        offsetListRight = []

        boxOffsetListLeft = []
        boxOffsetListRight = []


        if vdatumYslider.value() > 180:
            autoPilot = True
        else:
            autoPilot = False


        # box
        if 1 == 1:

            for i in range(len(lineBoxPoint)):
                BoxMove = 0
                tryCount = 0
                while True:
                    tryCount += 1
                    box = lineBoxPoint[i]
                    boxImg = modelOutput[box[1][1]:box[0][1] + 1,box[0][0]+BoxMove:box[1][0]+BoxMove  + 1]
                    if 0 in boxImg.shape:
                        #print(f"Error in box{i}: Box shape is (0, 0).")
                        break
                    # 將黑色顏色轉換為 NumPy 陣列
                    black_color = np.array(colors[0])

                    left_half = boxImg[:, :boxImg.shape[1] // 2, :]
                    right_half = boxImg[:, boxImg.shape[1] // 2:, :]
                    left_black_pixels = np.count_nonzero(np.all(left_half == black_color, axis=-1))
                    total_count = left_half.shape[0] * left_half.shape[1]
                    left_black_percent = left_black_pixels / total_count * 100
                    right_black_pixels = np.count_nonzero(np.all(right_half == black_color, axis=-1))
                    right_black_percent = right_black_pixels / total_count * 100

                    if left_black_percent + right_black_percent == 0 or tryCount > 15:
                        break
                    elif left_black_percent > right_black_percent:
                        BoxMove += 20
                    else:
                        BoxMove -= 20

                
                if tryCount > 15:
                    BoxMove = 0
                    continue
                
                # 在modelOutput往少的方向平移 20 個像素 直到% ==0

                boxOffsetListLeft.append((box[0][0]+BoxMove,int((box[0][1]+box[1][1])/2)))
                boxOffsetListRight.append((box[1][0]+BoxMove,int((box[0][1]+box[1][1])/2)))

                #劃出移動後的box
                #
                # 
                cv2.rectangle(frame_blend, (box[0][0]+BoxMove, box[0][1]), (box[1][0]+BoxMove, box[1][1]), (0, 255, 0), 2)

                # 在圖像上顯示黑色像素的百分比
                #cv2.putText(frame_blend, f"{left_black_percent:.2f}%,{right_black_percent:.2f}%", (box[0][0], box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



        leftOffsetAv = 0
        if 0 == 0:

            turnLeftPoint = []
            

            # turn left box
            for box in turnLeftBoxPoint:

                #往左移動直到黑色站超過一半
                BoxMove = 0
                tryCount = 0
                while True:
                    tryCount += 1
                    boxImg = modelOutput[box[1][1]:box[0][1] + 1,box[0][0]+BoxMove:box[1][0]+BoxMove  + 1]
                    if 0 in boxImg.shape:
                        #print(f"Error in box{i}: Box shape is (0, 0).")
                        break
                    # 將黑色顏色轉換為 NumPy 陣列
                    black_color = np.array(colors[0])
                    blackIndex = np.count_nonzero(np.all(boxImg == black_color, axis=-1))
                    total_count = boxImg.shape[0] * boxImg.shape[1]
                    black_percent = blackIndex / total_count

                    if black_percent >0.2 or tryCount > 15:
                        break
                    else:
                        BoxMove -= 20

                turnLeftPoint.append((box[0][0]+BoxMove+40,int((box[0][1]+box[1][1])/2)))
                #cv2.rectangle(frame_blend, (box[0][0]+BoxMove, box[0][1]), (box[1][0]+BoxMove, box[1][1]), (0, 255, 0), 2)

                leftOffsetAv += box[0][0]+BoxMove+40
            cv2.polylines(frame_blend, [np.array(turnLeftPoint)], False, (0, 255, 0), 2)

        rightOffsetAv = 0
        
        if 2== 2:
            turnRightPoint = []
            # turn right box
            for box in turnRightBoxPoint:

                #往右移動直到黑色站超過一半
                BoxMove = 0
                tryCount = 0
                while True:
                    tryCount += 1
                    boxImg = modelOutput[box[1][1]:box[0][1] + 1,box[0][0]+BoxMove:box[1][0]+BoxMove  + 1]
                    if 0 in boxImg.shape:
                        print(f"Error in box{i}: Box shape is (0, 0).")
                        break
                    # 將黑色顏色轉換為 NumPy 陣列
                    black_color = np.array(colors[0])
                    blackIndex = np.count_nonzero(np.all(boxImg == black_color, axis=-1))
                    total_count = boxImg.shape[0] * boxImg.shape[1]
                    black_percent = blackIndex / total_count

                    if black_percent >0.1 or tryCount > 15:
                        break
                    else:
                        BoxMove += 20

                turnRightPoint.append((box[1][0]+BoxMove-40,int((box[0][1]+box[1][1])/2)))
                #cv2.rectangle(frame_blend, (box[0][0]+BoxMove, box[0][1]), (box[1][0]+BoxMove, box[1][1]), (0, 255, 0), 2)
                rightOffsetAv += box[1][0]+BoxMove-40

            #劃出移動後的box
            cv2.polylines(frame_blend, [np.array(turnRightPoint)], False, (0, 255, 0), 2)
        
        allRoad = []

        if leftOffsetAv/len(turnLeftBoxPoint) < 150:
            allRoad.append("left")
        if rightOffsetAv/len(turnRightBoxPoint) > 700:
            allRoad.append("right")

        k = 1.5
        turnAngle = 0
        if datumYslider.value() == 1:
            
            try:
                oSum = 0
                for i in boxOffsetListLeft:
                    oSum += i[0]
                for i in boxOffsetListRight:
                    oSum += i[0]
                oSum = oSum/(len(boxOffsetListRight)+len(boxOffsetListLeft))
            except:
                oSum = 432
            #oSum = ((rightOffsetAv/len(turnRightBoxPoint)) + (leftOffsetAv/len(turnLeftBoxPoint)))/2
            
            turnAngle = map_range(oSum, 0, 864, -90*k, 90*k)

        elif datumYslider.value() == 2:
            
            if 'right' in allRoad:
                turnAngle = (rightOffsetAv/len(turnRightBoxPoint)-432)/2

            else:
                try:
                    oSum = 0
                    for i in boxOffsetListLeft:
                        oSum += i[0]
                    for i in boxOffsetListRight:
                        oSum += i[0]
                    oSum = oSum/(len(boxOffsetListRight)+len(boxOffsetListLeft))
                except:
                    oSum = 432
                turnAngle = map_range(oSum, 0, 864, -90*k, 90*k)


        elif datumYslider.value() == 0:

            if 'left' in allRoad:
                turnAngle = (leftOffsetAv/len(turnLeftBoxPoint) -432)/2
            else:
                try:
                    oSum = 0
                    for i in boxOffsetListLeft:
                        oSum += i[0]
                    for i in boxOffsetListRight:
                        oSum += i[0]
                    oSum = oSum/(len(boxOffsetListRight)+len(boxOffsetListLeft))
                except:
                    oSum = 432
                turnAngle = map_range(oSum, 0, 864, -90*k, 90*k)








        cv2.line(frame_blend, (432, 470), (432+int(turnAngle), 300), (0, 255, 0), 2)
        # cv2.polylines(frame_blend, [np.array(boxOffsetListLeft)], False, (0, 255, 0), 2)
        # cv2.polylines(frame_blend, [np.array(boxOffsetListRight)], False, (0, 255, 0), 2)
        
        displayAngle = turnAngle  
        turnAngle = map_range(turnAngle, -180, 180, 0, 180)

        if not autoPilot:
            turnAngle = vdatumYslider.value()


        # 空心圓
        cv2.circle(frame_blend, (80,380), 45, (255, 255, 255), 2)
        # 根據turnAngle角度 在圓上畫線
        cv2.line(frame_blend, (80,380), (80+int(math.cos(math.radians(turnAngle*2))*45),380+int(math.sin(math.radians(turnAngle*2))*45)), (255, 255, 255), 2)
        # turnAngle + 180
        cv2.line(frame_blend, (80,380), (80+int(math.cos(math.radians(turnAngle*2+180))*45),380+int(math.sin(math.radians(turnAngle*2+180))*45)), (255, 255, 255), 2)
        # turnAngle + 90
        cv2.line(frame_blend, (80,380), (80+int(math.cos(math.radians(turnAngle*2+90))*45),380+int(math.sin(math.radians(turnAngle*2+90))*45)),(255, 255, 255), 2)
        
        cv2.putText(frame_blend, f"Angle: {turnAngle:.2f}", (20,320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (00, 255, 0), 1)

        # open cv
        bytesPerline_blend = channel * width
        img_blend = QImage(frame_blend.data, width, height, bytesPerline_blend, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img_blend))


        # 开始绘制
        painter = QPainter(label.pixmap())
        font = QFont()
        font.setPointSize(15)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))  # 文字顏色，白色
        painter.drawText(20, 30, f"FPS: {fps}")  # 在左上角顯示FPS小數點後兩位
        painter.drawText(20, 50, f"Road: {0}")
        painter.drawText(20, 70, f"Angle: {turnAngle:.2f}")


        painter.drawText(20, 120, f"Latitude: {shared_gps_data['latitude']}")
        painter.drawText(20, 140, f"Longitude: {shared_gps_data['longitude']}")
        painter.drawText(20, 160, f"Site: {shared_gps_data['site']}")
        painter.drawText(20, 180, f"LoraState: {shared_gps_data['loraState']}")

        painter.drawText(20, 240, f"LeftOffset: {leftOffsetAv/len(turnLeftBoxPoint)}")
        painter.drawText(20, 260, f"RightOffset: {rightOffsetAv/len(turnRightBoxPoint)}")

        if autoPilot:
            painter.setPen(QColor(0, 255, 0))
            painter.drawText(200, 300, f"AutoPilot: ON")
        else:
            painter.setPen(QColor(255, 0, 0))
            painter.drawText(200, 30, f"AutoPilot: OFF")
        

        painter.setPen(QColor(255, 255, 255))

        painter.drawText(250, 30, f"servo:{servoState}")

        font.setPointSize(25)
        painter.setFont(font)
        painter.setPen(QColor(255, 0, 0))  # 文字顏色，白色

        if leftOffsetAv/len(turnLeftBoxPoint) < 150:
            painter.drawText(380,200, f"<<-")
        if rightOffsetAv/len(turnRightBoxPoint) > 700:
            painter.drawText(420, 200, f"->>")

        # 结束绘制
        painter.end()

        

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        #print("fps= %.2f"%(fps), end='\r')

        if openSerial:
            global ser
            ser.write((str(int(turnAngle))+'\n').encode())

        c= cv2.waitKey(1) & 0xff

        if video_save_path!="":
            out.write(frame)

        if c == ord('p') and c == 2:
            print("按下了 CTRL+P")
            autoPilot = not autoPilot

        if c==27:
            capture.release()
            break

video = threading.Thread(target=opencv)
video.start()

MainWindow.keyPressEvent = keyPressEvent
MainWindow.show()
sys.exit(app.exec_())