#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response
import cv2
import numpy as np
from PIL import Image
from deeplab import DeeplabV3

app = Flask(__name__)
# camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture("/Volumes/yihuanMissd/missd1t/IMG_1463.MOV")
deeplab = DeeplabV3()

speed = 20
mode = "original"

def gen_frames():


    while True:
        
        for i in range(speed):
            success, frame = camera.read()
        
        if not success:
            break
        else:
                
            if mode == "color" or mode == "sliding":
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                processed_frame = deeplab.detect_image(pil_frame)
                processed_frame = np.array(processed_frame)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

                if mode == "sliding":
                    pass


            if mode == "original":
                processed_frame = frame

            ret, buffer = cv2.imencode('.jpg', processed_frame)
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


if __name__ == '__main__':
    app.run('0.0.0.0', port=5001)
