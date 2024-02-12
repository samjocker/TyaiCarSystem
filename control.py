from view import get_view
import cv2
import numpy as np

cap = cv2.VideoCapture("E:\\missd1t\\test5.mp4")



def get_edges(pil_image,original_image):




    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    original_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200) 


    kernel = np.ones((5,5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)  

    kernel = np.ones((5,5), np.uint8)
    eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1) 


    _, thresh = cv2.threshold(dilated_edges, 0, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    min_area_threshold = 5000

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area_threshold:
            thresh[labels == label] = 0

    # 霍夫直線變換
    lines = cv2.HoughLines(thresh, 1, np.pi/180, threshold=300)  # 可根據需要調整閾值


    # 繪製直線段
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)  # 將弧度轉換為角度
            if 45 <= angle <= 135:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(original_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                

    return original_image
while True:

    for i in range(20): 
        ret, frame = cap.read()

    frame = cv2.resize(frame, (864, 480))
    display_frame = frame.copy()


    if ret:
        processed_frame = get_view(frame)

        display_frame = get_edges(processed_frame[2],processed_frame[0])

        cv2.imshow('Processed Frame', np.array(display_frame))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break