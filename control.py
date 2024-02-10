from view import get_view
import cv2
import numpy as np

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (864, 480))

    if ret:
        processed_frame = get_view(frame)
        cv2.imshow('Processed Frame', np.array(processed_frame[0]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break