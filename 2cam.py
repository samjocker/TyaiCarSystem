import cv2

# 讀取相機的視頻流
cap = cv2.VideoCapture(0)  # 如果是外部相機，可能需要修改引數
cap2 = cv2.VideoCapture(1)

while True:
    # 從相機中讀取視頻幀
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    if not ret:
        break

    # 顯示畫面
    cv2.imshow('Camera Feed', frame)
    cv2.imshow('Camera Feed2', frame2)

    # 按下'q'鍵退出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
