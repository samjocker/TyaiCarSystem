import cv2
from flask import Flask, render_template, Response
import threading

app = Flask(__name__)

def video_generator():
    while True:
        # 擷取影像
        ret, frame = cap.read()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 將影像轉換為 bytes 格式
        frame_bytes = cv2.imencode('.jpg', rgb_frame)[1].tobytes()

        yield frame_bytes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(video_generator(), mimetype='multipart/x-frames')


if __name__ == '__main__':
    # 開啟視訊鏡頭
    cap = cv2.VideoCapture(0)

    # 建立 OpenCV 線程
    video_thread = threading.Thread(target=video_generator)
    video_thread.daemon = True
    video_thread.start()

    # 啟動 Flask 應用程式
    app.run(debug=True)
