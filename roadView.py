import cv2
import numpy as np
import base64
import requests
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

class DisplayApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("Display App")

        self.label = QLabel(self)
        self.label.setGeometry(10, 10, 640, 480)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(30)  # 更新頻率，單位是毫秒

    def update_display(self):
        # 從攝像頭獲取影像
        _, frame = self.capture.read()

        # 將影像轉換為 base64 字串
        _, img_encoded = cv2.imencode('.jpg', frame)
        image_data = {'image': base64.b64encode(img_encoded).decode('utf-8')}

        # 向 AI 運算端發送 POST 請求並獲取結果
        try:
            response = requests.post('127.0.0.1:5001/process_image', data=image_data)
            result_data = response.json()

            # 解碼 base64 影像資料
            result_image_np = np.array(result_data['result_image'], dtype=np.uint8)
            result_image = cv2.cvtColor(result_image_np, cv2.COLOR_RGB2BGR)

            # 顯示結果
            result_image = cv2.resize(result_image, (640, 480))
            height, width, channel = result_image.shape
            bytesPerline = 3 * width
            q_image = QImage(result_image.data, width, height, bytesPerline, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.label.setPixmap(pixmap)

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to AI server: {e}")

    def closeEvent(self, event):
        self.capture.release()
        event.accept()

    def start_capture(self):
        self.capture = cv2.VideoCapture(0)

if __name__ == '__main__':
    app = QApplication([])
    display_app = DisplayApp()
    display_app.start_capture()
    display_app.show()
    app.exec_()
