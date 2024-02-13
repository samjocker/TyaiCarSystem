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
# latitude, longitude = 24.99250, 121.32032

if latitude > 50:
    temp = longitude
    longitude = latitude
    latitude = temp

# 獲取地圖數據
G = ox.graph_from_point((latitude, longitude), dist=400, network_type='drive_service')
print(G)
origin = ox.distance.nearest_nodes(G, longitude, latitude)
destination = ox.distance.nearest_nodes(G, 121.32012, 24.99422)

route = nx.shortest_path(G, origin, destination)
# ox.plot_graph_route(G, route)

# 取得所有點的座標
nodes = list(G.nodes())
node_coordinates = []
smallestCdn = [122.0, 25.0]
biggestCdn = [0.0, 0.0]

for node in nodes:
    x, y = G.nodes[node]['x'], G.nodes[node]['y']
    node_coordinates.append((x, y))


# 取得所有線條的座標
edges = list(G.edges())
edge_coordinates = []
for edge in edges:
    u, v = edge
    x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
    x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
    edge_coordinates.append([(x1, y1), (x2, y2)])

# print("所有點的座標：", node_coordinates)
# print("所有線條的座標：", edge_coordinates)
    
# 計算最右最左最上最下的值
min_x = min([x for x, _ in node_coordinates])
min_y = min([y for _, y in node_coordinates])
max_x = max([x for x, _ in node_coordinates])
max_y = max([y for _, y in node_coordinates])

# 等比例轉換成能放進去500*500的OpenCV空白畫面內顯示
width = 500
height = 500
scale_x = width / (max_x - min_x)
scale_y = height / (max_y - min_y)

# 繪製地圖
img = np.zeros((height, width, 3), dtype=np.uint8)
img.fill(255)
for u, v in edge_coordinates:
    ux = int((u[0] - min_x) * scale_x)
    uy = int((u[1] - min_y) * scale_y)
    vx = int((v[0] - min_x) * scale_x)
    vy = int((v[1] - min_y) * scale_y)
    cv2.line(img, (ux, uy), (vx, vy), (248, 242, 241), 4)

for x, y in node_coordinates:
    x = int((x - min_x) * scale_x - 1)
    y = int((y - min_y) * scale_y - 1)
    # img[y, x] = (255, 255, 255)
    cv2.circle(img, (x, y), 2, (126, 201, 255), 2, -1)

cv2.circle(img, (int((longitude - min_x) * scale_x - 1), int((latitude - min_y) * scale_y - 1)), 3, (255, 205, 125), 3, -1)
img = cv2.flip(img, 0)
cv2.imshow('Map', img)
cv2.waitKey(0)

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
# cv2.imshow('Adjusted Map', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
