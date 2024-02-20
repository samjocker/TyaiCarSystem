
import time
import cv2
import numpy as np
from PIL import Image
from deeplab import DeeplabV3
import math

notRoadColors = [(0, 128, 0), (0, 128, 128), (128, 0, 0), (128, 0, 128), (128, 128, 0),
          (128, 128, 128)]

deeplab = DeeplabV3()

def perspective_correction(image,ups,dos,viewup):
    original_points = np.float32([[0-dos, 480], [720+dos, 480],[ups, viewup], [720-ups, viewup]])
    corrected_points = np.float32([[200, 720], [520, 720], [200, 0], [520, 0]])
    perspective_matrix = cv2.getPerspectiveTransform(original_points, corrected_points)
    result = cv2.warpPerspective(image, perspective_matrix, (720, 720))
    return result

def inverse_perspective_correction(image, ups, dos, viewup):
    corrected_points = np.float32([[0-dos, 480], [720+dos, 480],[ups, viewup], [720-ups, viewup]])
    original_points = np.float32([[200, 720], [520, 720], [200, 0], [520, 0]])
    inverse_perspective_matrix = cv2.getPerspectiveTransform(original_points, corrected_points)
    result = cv2.warpPerspective(image, inverse_perspective_matrix, (720, 480))
    return result

if __name__ == "__main__":
    #video_path = r"D:\Data\project\tyaiCar\TyaiCarSystem\IMG_1461.MOV"
    video_path = '/media/yihuan/hdd/Data/project/tyaiCar/TyaiCarSystem/IMG_1461.MOV'
    video_save_path = ""
    video_fps = 25.0

    capture = cv2.VideoCapture(video_path)
    if video_save_path != "":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)


    fps = 0.0
    video_speed = 1

    def nothing(x):
        pass
    
    box_display_on = False
    real_mode = False
    setMode = False

    # 创建窗口
    cv2.namedWindow("Parameters")
    #設定大小
    cv2.resizeWindow("Parameters", 400, 700)

    cv2.createTrackbar("pc_up", "Parameters", 300, 300, nothing)
    cv2.createTrackbar("pc_down", "Parameters", 96, 300, nothing)
    cv2.createTrackbar("pc_view", "Parameters", 216, 300, nothing)
    cv2.createTrackbar("box_width", "Parameters", 120, 300, nothing)
    cv2.createTrackbar("tryLong", "Parameters", 400, 400, nothing)
    cv2.createTrackbar("box_display", "Parameters", 0, 1, nothing)
    cv2.createTrackbar("realMode", "Parameters", 0, 1, nothing)

    cv2.createTrackbar("car_width", "Parameters", 70, 300, nothing)

    cv2.createTrackbar("repc_up", "Parameters", 300, 300, nothing)
    cv2.createTrackbar("repc_down", "Parameters", 0, 300, nothing)
    cv2.createTrackbar("repc_view", "Parameters", 100, 300, nothing)

    cv2.createTrackbar("videoSpeed", "Parameters", 1, 30, nothing)

    cv2.createTrackbar("setMode", "Parameters", 0, 1, nothing)



    while True:
        
        t1 = time.time()
        
        for i in range(video_speed):
            ref, frame = capture.read()

        if not ref:
            break
        
        original_frame = frame.copy()

        frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        frame = np.array(deeplab.detect_image(frame))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        mask = (frame == [0,0,128]).all(axis=2)
        frame[mask] = [255, 255, 255]

        for color in notRoadColors:
            color = np.array(color)
            mask = (frame == color).all(axis=2)
            frame[mask] = [0, 0, 0]


        perspective_correction_up = cv2.getTrackbarPos("pc_up", "Parameters")
        perspective_correction_down = cv2.getTrackbarPos("pc_down", "Parameters")
        perspective_correction_view = cv2.getTrackbarPos("pc_view", "Parameters")
        box_width = cv2.getTrackbarPos("box_width", "Parameters")
        tryLong = cv2.getTrackbarPos("tryLong", "Parameters")
        box_display_on = cv2.getTrackbarPos("box_display", "Parameters")
        car_width = cv2.getTrackbarPos("car_width", "Parameters")
        repc_up = cv2.getTrackbarPos("repc_up", "Parameters")
        repc_down = cv2.getTrackbarPos("repc_down", "Parameters")
        repc_view = cv2.getTrackbarPos("repc_view", "Parameters")
        real_mode = cv2.getTrackbarPos("realMode", "Parameters")
        video_speed = cv2.getTrackbarPos("videoSpeed", "Parameters")
        setMode = cv2.getTrackbarPos("setMode", "Parameters")


        frame = perspective_correction(frame, perspective_correction_up, perspective_correction_down, perspective_correction_view)
        
        if real_mode:
            original_frame = cv2.resize(original_frame, (720, 480), interpolation=cv2.INTER_LINEAR)
            original_frame = perspective_correction(original_frame, perspective_correction_up, perspective_correction_down, perspective_correction_view)

        canny_frame = cv2.Canny(frame, threshold1=100, threshold2=25)

        kernel = np.ones((5,5), np.uint8)
        canny_frame = cv2.dilate(canny_frame, kernel, iterations=3)
        # frame = cv2.erode(frame, kernel, iterations=2)



        #print(frame.shape, canny_frame.shape)
        canny_frame_with_channels = np.expand_dims(canny_frame, axis=-1)
        canny_frame_with_channels = np.repeat(canny_frame_with_channels, 3, axis=-1)
        frame = cv2.addWeighted(frame, 1, canny_frame_with_channels, 1, 0.5)

        mask = (frame == [0,0,0]).all(axis=2)
        frame[mask] = [200, 200, 200]


        #將畫面轉為陣列   [200, 200, 200]=0   [255, 255, 255]=1
        binary_array = (frame == [255, 255, 255]).all(axis=-1).astype(np.uint8)

        if real_mode:
            frame = original_frame

        # 滑動箱子

        roadPoints = []

        ## 直走模式
        for i in range(719, 719-tryLong,-20):
            
            box_move = 0
            left_total_sum = 0
            right_total_sum = 0
            boxLeft = []
            boxRight = []

            for tryCount in range(20):

                boxLeft = [[360-box_width+box_move,i],[360+box_move,i-20]]

                region = binary_array[boxLeft[1][1]:boxLeft[0][1], boxLeft[0][0]:boxLeft[1][0]]
                left_total_sum = np.sum(region)

                boxRight = [[360+box_move,i],[360+box_width+box_move,i-20]]
                region = binary_array[boxRight[1][1]:boxRight[0][1], boxRight[0][0]:boxRight[1][0]]
                right_total_sum = np.sum(region)

                if left_total_sum >  right_total_sum :
                    box_move -= 20
                elif left_total_sum <  right_total_sum :
                    box_move += 20
                else:
                    break
            
            if left_total_sum != 0 and right_total_sum != 0:
                
                roadPoints.append([360+box_move, i])

                if box_display_on:
                    frame = cv2.rectangle(frame, (boxLeft[0][0],boxLeft[0][1]),(boxLeft[1][0],boxLeft[1][1]), (0, 255, 0), 2)
                    #frame = cv2.putText(frame, "l= %.2f"%(left_total_sum), (boxLeft[0][0],boxLeft[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    frame = cv2.rectangle(frame, (boxRight[0][0],boxRight[0][1]),(boxRight[1][0],boxRight[1][1]), (0, 255, 0), 2)
                    #frame = cv2.putText(frame, "r= %.2f"%(right_total_sum), (boxRight[0][0],boxRight[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        #cv2.polylines(frame, [np.array(roadPoints)], False, (0, 255, 0), 15)
        ### 直走模式結尾
              
        # ## 右轉模式
        # for i in range(719, 719-tryLong,-20):
            
        #     box_move = 0
        #     total_sum = 0
        #     box = []

        #     for tryCount in range(20):

        #         box = [[360-int(box_width)+box_move,i],[360+int(box_width)+box_move,i-20]]
        #         region = binary_array[box[1][1]:box[0][1], box[0][0]:box[1][0]]
        #         total_sum = np.sum(region)

        #         if total_sum < box_width*20:
        #             break
        #         else:
        #             box_move += 20
                
        #     roadPoints.append([360+box_move+int(box_width), i])

        #     if box_display_on:
        #         frame = cv2.rectangle(frame, (box[0][0],box[0][1]),(box[1][0],box[1][1]), (0, 255, 0), 2)
        #         #frame = cv2.putText(frame, "l= %.2f"%(left_total_sum), (boxLeft[0][0],boxLeft[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #         #frame = cv2.putText(frame, "r= %.2f"%(right_total_sum), (boxRight[0][0],boxRight[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        # #cv2.polylines(frame, [np.array(roadPoints)], False, (0, 255, 0), 15)




        newLine = []
        for i in range(0,len(roadPoints)-1,2):
            #兩點平均值
            newLine.append([int((roadPoints[i][0]+roadPoints[i+1][0])/2),int((roadPoints[i][1]+roadPoints[i+1][1])/2)])


        full_line_Left = []
        full_line_Right = []

        for i in range(0,len(newLine)-1,1):

            if newLine[i][0] == newLine[i+1][0]:
                #frame = cv2.line(frame, (int(newLine[i+1][0]+car_width/2), newLine[i][1]), (int(newLine[i+1][0]-car_width/2), newLine[i][1]), (0, 255, 0), 2)
                full_line_Left.append([int(newLine[i+1][0]+car_width/2), newLine[i][1]])
                full_line_Right.append([int(newLine[i+1][0]-car_width/2), newLine[i][1]])
            else:

                slope = (newLine[i+1][1] - newLine[i][1]) / (newLine[i+1][0] - newLine[i][0])
                perpendicular_slope = -1 / slope  # 垂直於原斜率的斜率
                center_x = newLine[i+1][0]
                center_y = newLine[i+1][1]
                half_width = car_width / 2

                dx = np.sqrt(half_width**2 / (1 + perpendicular_slope**2))
                x1 = int(center_x + dx)
                y1 = int(center_y + perpendicular_slope * dx)
                x2 = int(center_x - dx)
                y2 = int(center_y - perpendicular_slope * dx)

                #frame = cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                full_line_Left.append([x1, y1])
                full_line_Right.append([x2, y2])

        full_line_Left = []
        full_line_Right = []

        full_line_Left.append([int(360+car_width/2), 720])
        full_line_Right.append([int(360-car_width/2), 720])

        roadPoints = []

        for i in range(719, 719-tryLong,-20):
            
            box_move = 0
            left_total_sum = 0
            right_total_sum = 0
            boxLeft = []
            boxRight = []

            for tryCount in range(20):

                boxLeft = [[360-box_width+box_move,i],[360+box_move,i-20]]

                region = binary_array[boxLeft[1][1]:boxLeft[0][1], boxLeft[0][0]:boxLeft[1][0]]
                left_total_sum = np.sum(region)

                boxRight = [[360+box_move,i],[360+box_width+box_move,i-20]]
                region = binary_array[boxRight[1][1]:boxRight[0][1], boxRight[0][0]:boxRight[1][0]]
                right_total_sum = np.sum(region)

                if left_total_sum >  right_total_sum :
                    box_move -= 20
                elif left_total_sum <  right_total_sum :
                    box_move += 20
                else:
                    break
            
            if left_total_sum != 0 and right_total_sum != 0:
                
                roadPoints.append([360+box_move, i])

                if box_display_on:
                    frame = cv2.rectangle(frame, (boxLeft[0][0],boxLeft[0][1]),(boxLeft[1][0],boxLeft[1][1]), (0, 255, 0), 2)
                    #frame = cv2.putText(frame, "l= %.2f"%(left_total_sum), (boxLeft[0][0],boxLeft[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    frame = cv2.rectangle(frame, (boxRight[0][0],boxRight[0][1]),(boxRight[1][0],boxRight[1][1]), (0, 255, 0), 2)
                    #frame = cv2.putText(frame, "r= %.2f"%(right_total_sum), (boxRight[0][0],boxRight[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                break
        #cv2.polylines(frame, [np.array(roadPoints)], False, (0, 255, 0), 15)
                        
        newLine = []
        for i in range(0,len(roadPoints)-1,2):
            #兩點平均值
            newLine.append([int((roadPoints[i][0]+roadPoints[i+1][0])/2),int((roadPoints[i][1]+roadPoints[i+1][1])/2)])


        for i in range(0,len(newLine)-1,1):

            if newLine[i][0] == newLine[i+1][0]:
                #frame = cv2.line(frame, (int(newLine[i+1][0]+car_width/2), newLine[i][1]), (int(newLine[i+1][0]-car_width/2), newLine[i][1]), (0, 255, 0), 2)
                full_line_Left.append([int(newLine[i+1][0]+car_width/2), newLine[i][1]])
                full_line_Right.append([int(newLine[i+1][0]-car_width/2), newLine[i][1]])
            else:

                slope = (newLine[i+1][1] - newLine[i][1]) / (newLine[i+1][0] - newLine[i][0])
                perpendicular_slope = -1 / slope  # 垂直於原斜率的斜率
                center_x = newLine[i+1][0]
                center_y = newLine[i+1][1]
                half_width = car_width / 2

                dx = np.sqrt(half_width**2 / (1 + perpendicular_slope**2))
                x1 = int(center_x + dx)
                y1 = int(center_y + perpendicular_slope * dx)
                x2 = int(center_x - dx)
                y2 = int(center_y - perpendicular_slope * dx)

                #frame = cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                full_line_Left.append([x1, y1])
                full_line_Right.append([x2, y2])


        full_line_Left = np.array(full_line_Left)
        full_line_Right = np.array(full_line_Right)


        fill_color = (128, 255, 0)

        frame = cv2.polylines(frame, [full_line_Left], False, fill_color, 2)
        frame = cv2.polylines(frame, [full_line_Right], False, fill_color, 2)
                        
        fill_polygon = np.concatenate((full_line_Left, np.flip(full_line_Right, axis=0)), axis=0)
        frame = cv2.fillPoly(frame, [fill_polygon], fill_color)
                                    
        
        if not setMode:
            frame = inverse_perspective_correction(frame, repc_up, repc_down, repc_view)

            mask = (frame == [0,0,0]).all(axis=2)
            frame[mask] = [220, 220, 220]


        fps = (fps + (1./(time.time()-t1))) / 2
        print("fps= %.2f"%(fps), end="\r")

        # fps 顯示
        #frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        cv2.imshow("video", frame)
        c = cv2.waitKey(1) & 0xff
        if video_save_path != "":
            out.write(frame)
        if c == 27:
            capture.release()
            break


    print("Video Detection Done!")
    capture.release()

    if video_save_path != "":
        print("Save processed video to the path :" + video_save_path)
        out.release()

    cv2.destroyAllWindows()
