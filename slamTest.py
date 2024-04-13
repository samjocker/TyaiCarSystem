import numpy as np
import cv2 as cv

def find_feature_matches(img_1, img_2):
    """寻找特征匹配的点

    Args:
        img_1: pass
        img_2: pass

    Returns:
        kp1: 
        kp2: 
        good_match: 

    """
    orb = cv.ORB_create()

    kp1 = orb.detect(img_1)
    kp2 = orb.detect(img_2)

    kp1, des1 = orb.compute(img_1, kp1)
    kp2, des2 = orb.compute(img_2, kp2)

    bf = cv.BFMatcher(cv.NORM_HAMMING)

    matches = bf.match(des1, des2)

    min_distance = matches[0].distance
    max_distance = matches[0].distance

    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance
    
    print("Max dist:", max_distance)
    print("Min dist:", min_distance)

    good_match = []

    for x in matches:
        if x.distance <= max(2*min_distance, 30.0):
            good_match.append(x)

    return kp1, kp2, good_match


if __name__ == "__main__":
    cap = cv.VideoCapture('/Users/sam/Documents/MyProject/mixProject/TYAIcar/MLtraning/visualIdentityVideo/IMG_1420.MOV')  # 設置影片路徑
    
    if not cap.isOpened():
        print("Error: Unable to open video.")
        exit()
    
    ret, prev_frame = cap.read()
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    
    # 設置深度圖像路徑
    depth_file = '/Users/sam/Documents/MyProject/mixProject/TYAIcar/MLtraning/visualIdentityVideo/IMG_1420.MOV'
    depth_cap = cv.VideoCapture(depth_file)
    
    if not depth_cap.isOpened():
        print("Error: Unable to open depth video.")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # 讀取深度圖像
        ret_depth, depth_frame = depth_cap.read()
        if not ret_depth:
            break
        
        depth_gray = cv.cvtColor(depth_frame, cv.COLOR_BGR2GRAY)

        # 图像匹配
        keypoints_1, keypoints_2, matches = find_feature_matches(prev_gray, gray)
        print("共计匹配点:", len(matches))

        # 筛选特征点
        pts2 = []
        pts1 = []
        for i in range(int(len(matches))):
            pts1.append(keypoints_1[matches[i].queryIdx].pt)
            pts2.append(keypoints_2[matches[i].trainIdx].pt)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        # 建立3D点
        K=[[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]]
        K=np.array(K)
        pts_3d = []
        pts_2d = []
        for i in range(pts1.shape[0]): 
            p1 = pts1[i]
            d1 = depth_gray[int(p1[1]),int(p1[0])]/1000.0
            if d1 == 0: 
                continue
            p1 = (p1 - (K[0][2],K[1][2]))/(K[0][0],K[1][1])*d1 
            pts_3d.append([p1[0], p1[1], d1])

        print("最终匹配数：", len(pts_3d))
        pts_3d = np.float64(pts_3d)
        pts_2d = np.float64(pts2)
        print("3D点：")
        print(pts_3d)

        flag,R,t = cv.solvePnP(pts_3d,pts_2d,K,None)
        R,Jacobian  = cv.Rodrigues(R)
        print("旋转矩阵R：\n", R)
        print("平移矩阵t：\n", t)
        
        prev_gray = gray

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    depth_cap.release()
    cv.destroyAllWindows()
