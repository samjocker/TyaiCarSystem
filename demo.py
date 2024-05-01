
import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from deeplab import DeeplabV3



def gen_frames():

    deeplab = DeeplabV3()
    mode = "video"

    video_path      = 0
    video_save_path = ""
    video_fps       = 30

    play_speed = 1

    
    capture=cv2.VideoCapture(video_path)
    if video_save_path!="":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    ref, frame = capture.read()
    if not ref:
        raise ValueError("鏡頭連接失敗")

    fps = 0.0
    while(True):

        t1 = time.time()

        for i in range(play_speed):
            ref, frame = capture.read()

        if not ref:
            break

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        frame = np.array(deeplab.detect_image(frame))
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %.2f"%(fps))
        frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("video",frame)
        c= cv2.waitKey(1) & 0xff 
        if video_save_path!="":
            out.write(frame)

        if c==27:
            capture.release()
            break


    print("Video Detection Done!")
    capture.release()
    if video_save_path!="":
        print("Save processed video to the path :" + video_save_path)
        out.release()
        
    cv2.destroyAllWindows()