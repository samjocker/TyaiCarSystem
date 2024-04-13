import cv2
import os

# 指定資料夾路徑
folder_path = "road_corrected_images"

# 列出資料夾中的所有影像檔案
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

roadClass = ['straight', 'turn left', 'turn right', 'wide', 'T-intersection','crossRoad','right fork','left fork','none']
#                 0          1           2          3         4              5             6           7          8

# 顯示並接收輸入3333333333
for image_file in image_files:
    # 構建完整的檔案路徑
    image_path = os.path.join(folder_path, image_file)

    # 讀取影像
    img = cv2.imread(image_path)
    img2 = img.copy()
    img = cv2.resize(img, (30, 30))

    # 顯示影像
    cv2.imshow('Image', img2)
    
    # 接收輸入
    key = cv2.waitKey(0) & 0xFF

    # 檢查按鍵是否是0-9之間的數字
    if ord('0') <= key <= ord('9'):
        # 構建目標資料夾路徑
        target_folder = os.path.join('train\image', str(key))
        
        # 如果目標資料夾不存在，則創建
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # 將影像儲存到目標資料夾中
        target_path = os.path.join(target_folder, image_file)
        cv2.imwrite(target_path, img)

        print(f"影像已儲存到 {target_path}")

# 關閉視窗
cv2.destroyAllWindows()
