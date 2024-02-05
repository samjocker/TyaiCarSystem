import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import cv2
import numpy as np
import requests

# 定義取得座標資料的函數
def get_coordinates(api_url):
    response = requests.get(api_url)
    data = response.json()
    print(data)
    latitude = data["latitude"]
    longitude = data["longitude"]
    return latitude, longitude

# 取得座標資料
api_url = "http://xhinherpro.xamjiang.com/getData"
latitude, longitude = get_coordinates(api_url)

# 獲取地圖數據
G = ox.graph_from_point((longitude, latitude), dist=150, network_type='drive_service')

# 繪製地圖並設定路徑和背景的顏色
fig, ax = ox.plot_graph(G, show=False, close=False, figsize=(10, 10), edge_color='white', bgcolor='gray', edge_linewidth=10.0)

# 將Matplotlib圖像轉換為OpenCV格式
fig.canvas.draw()
img = np.array(fig.canvas.renderer.buffer_rgba())

# 對圖像進行顏色和對比度調整（可以根據需要進行更進一步的調整）
alpha = 1.5  # 控制對比度
beta = 30    # 控制亮度
adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# 顯示調整後的圖像
cv2.imshow('Adjusted Map', adjusted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
