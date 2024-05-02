from flask import Flask, Response, render_template
import cv2
import mss
import numpy as np
import time

app = Flask(__name__)
screen_capture = None

def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        return img

def generate_frames():
    global screen_capture
    while True:

        time.sleep(0.1)
        frame = capture_screen()
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global screen_capture
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run('0.0.0.0', port=5001,debug=True)
