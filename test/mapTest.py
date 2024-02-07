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
# latitude, longitude = get_coordinates(api_url)
latitude, longitude = 24.99250, 121.32032

if latitude > 50:
    temp = longitude
    longitude = latitude
    latitude = temp

# 獲取地圖數據
G = ox.graph_from_point((latitude, longitude), dist=400, network_type='drive_service')
origin = ox.distance.nearest_nodes(G, longitude, latitude)
destination = ox.distance.nearest_nodes(G, 121.32012, 24.99422)

route = nx.shortest_path(G, origin, destination)
ox.plot_graph_route(G, route)

# 繪製地圖並設定路徑和背景的顏色
fig, ax = ox.plot_graph(G, show=False, close=True, figsize=(10, 10), edge_color='#FFF', bgcolor='#F1F2F8', edge_linewidth=5.0, node_size=20, node_color="#FFC97E")

ax.scatter(latitude, longitude, c='green')

# 將Matplotlib圖像轉換為OpenCV格式
fig.canvas.draw()
img = np.array(fig.canvas.renderer.buffer_rgba())
img = cv2.merge([img[:, :, 2], img[:, :, 1], img[:, :, 0]])

cdnX = int(img.shape[0]/2)
cdnY = int(img.shape[1]/2)

# cv2.circle(img, (cdnX, cdnY), 12, (255, 255, 255), -1)
# cv2.circle(img, (cdnX, cdnY), 15, (0, 149, 255), 5)

# 顯示調整後的圖像
cv2.imshow('Adjusted Map', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
