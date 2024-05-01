#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response
import cv2
import numpy as np
from PIL import Image
from deeplab import DeeplabV3

app = Flask(__name__)
# camera = cv2.VideoCapture(0)

video_path = r"D:\Data\project\tyaiCar\TyaiCarSystem\IMG_1461.MOV"
camera = cv2.VideoCapture(video_path)
deeplab = DeeplabV3()

speed = 3
mode = "original"


AutoPilot = False
turnAngle = 0
carSpeed = 0

def gen_frames():


    while True:
        
        for i in range(speed):
            success, frame = camera.read()
        
        if not success:
            break
        else:
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            processed_frame,pr = deeplab.detect_image(pil_frame)
            processed_frame = np.array(processed_frame)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)


            print(pr[500,500])


            if mode == "original":

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            elif mode == "color":

                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            elif mode == "sliding":



                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/original_video_feed')
def original_video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def original_ndex():
    return render_template('original.html')

@app.route('/switch')
def switch_mode():
    global mode

    if mode == "original":
        mode = "color"
    elif mode == "color":
        mode = "sliding"
    elif mode == "sliding":
        mode = "original"

    return "sam is small"

@app.route('/status')
def status():
    global AutoPilot
    global turnAngle
    global carSpeed 

    return {
        "AutoPilot": AutoPilot,
        "turnAngle": turnAngle,
        "carSpeed": carSpeed
    }

if __name__ == '__main__':
    app.run('0.0.0.0', port=5001)
