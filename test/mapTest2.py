import xml.etree.ElementTree as ET
import osmnx as ox
import networkx as nx
import cv2
import numpy as np

# 讀取OSM檔案
osm_file_path = "TYAIcampus3.osm"
tree = ET.parse(osm_file_path)
root = tree.getroot()

# 創建節點字典以便根據ID查找座標
node_dict = {}
for node in root.findall('node'):
    node_id = node.attrib['id']
    latitude = float(node.attrib['lat'])
    longitude = float(node.attrib['lon'])
    node_dict[node_id] = (latitude, longitude)

# 創建way字典以便根據ID查找節點參考和標籤
way_dict = {}
for way in root.findall('way'):
    way_id = way.attrib['id']
    node_refs = [nd.attrib['ref'] for nd in way.findall('nd')]
    tags = {tag.attrib['k']: tag.attrib['v'] for tag in way.findall('tag')}
    way_dict[way_id] = {'node_refs': node_refs, 'tags': tags}

# 獲取路徑的範圍
min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf

# 遍歷所有way並僅顯示service等於driveway的路徑
for way_id, data in way_dict.items():
    if 'service' in data['tags'] and data['tags']['service'] == 'driveway':
        route = []
        # 從way中提取節點座標
        for node_id in data['node_refs']:
            if node_id in node_dict:
                route.append(node_dict[node_id])
        route = np.array(route)

        # 更新範圍
        min_x = min(min_x, np.min(route[:, 1]))
        max_x = max(max_x, np.max(route[:, 1]))
        min_y = min(min_y, np.min(route[:, 0]))
        max_y = max(max_y, np.max(route[:, 0]))

# 計算縮放比例
width = 500
height = 500
scale_x = width / (max_x - min_x)
scale_y = height / (max_y - min_y)

def convertCdn(num, type):
    if type == "x":
        return int((num - min_x) * scale_x)
    elif type == "y":
        return int((num - min_y) * scale_y)

# 創建空白畫布
img = np.zeros((height, width, 3), dtype=np.uint8)
img.fill(255)

# 繪製每個way的路徑並顯示節點
canGoWay = []
for way_id, data in way_dict.items():
    if 'service' in data['tags'] and data['tags']['service'] == 'driveway':
        route = []
        # 從way中提取節點座標
        for node_id in data['node_refs']:
            if node_id in node_dict:
                route.append(node_dict[node_id])
        route = np.array(route)
        
        # 繪製路徑
        for i in range(len(route) - 1):
            u = route[i]
            v = route[i + 1]
            ux = convertCdn(u[1], "x")
            uy = convertCdn(u[0], "y")
            vx = convertCdn(v[1], "x")
            vy = convertCdn(v[0], "y")
            cv2.line(img, (ux, uy), (vx, vy), (245, 232, 217), 8)
            try:
                site = data['tags']['target']
                canGoWay.append([(ux, uy), (vx, vy), site])
            except:
                pass
        
        # 繪製節點
        for node in route:
            x = int((node[1] - min_x) * scale_x - 1)
            y = int((node[0] - min_y) * scale_y - 1)
            cv2.circle(img, (x, y), 3, (212, 171, 134), -1)

myCdn = [121.32149, 24.99208]
myCdn = [convertCdn(myCdn[0], "x"), convertCdn(myCdn[1], "y")]

def point_to_line_distance(point, line):
    """
    計算點到直線的垂直距離
    :param point: 點的座標 (x, y)
    :param line: 直線的兩個端點 ((x1, y1), (x2, y2))
    :return: 點到直線的垂直距離
    """
    x, y = point
    x1, y1 = line[0]
    x2, y2 = line[1]
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        # 線段長度為0，返回點與端點之間的距離
        return np.linalg.norm(np.array(point) - np.array(line[0]))
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))  # 確保t在[0, 1]範圍內
    closest_point = (x1 + t * dx, y1 + t * dy)
    distance = np.linalg.norm(np.array(point) - np.array(closest_point))
    return distance

# 找出距離最近的線條
closest_way = None
closest_distance = np.inf
for way in canGoWay:
    u, v, site = way
    dist = point_to_line_distance(myCdn, (u, v))
    if dist < closest_distance:
        closest_way = way
        closest_distance = dist

# 打印最接近的線條和距離
if closest_way is not None:
    u, v, site = closest_way
    # cv2.line(img, u, v, (0, 255, 0), 4)
    print("Closest way:", u, "-", v, "Distance:", closest_distance, "Site:", site)
else:
    print("No closest way found.")


# 規劃路線
G = ox.graph_from_point((24.99250, 121.32032), dist=200, network_type='drive_service')

startX, startY = 121.32167, 24.99179
endX, endY = 121.31989, 24.99412

origin = ox.distance.nearest_nodes(G, startX, startY)
destination = ox.distance.nearest_nodes(G, endX, endY)

route = nx.shortest_path(G, origin, destination)
lastNode = (convertCdn(startX, "x"), convertCdn(startY, "y"))
routeList = []
for node in route:
    x, y = G.nodes[node]['x'], G.nodes[node]['y']
    x = convertCdn(x, "x")
    y = convertCdn(y, "y")
    cv2.line(img, lastNode, (x, y), (105, 66, 48), 3)
    routeList.append((x, y))
    lastNode = (x, y)
    # cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
cv2.line(img, lastNode, (convertCdn(endX, "x"), convertCdn(endY, "y")), (105, 66, 48), 3)
cv2.circle(img, (convertCdn(startX, "x"), convertCdn(startY, "y")), 6, (1, 97, 242), -1)
cv2.circle(img, (convertCdn(endX, "x"), convertCdn(endY, "y")), 4, (255, 255, 255), -1)
cv2.circle(img, (convertCdn(endX, "x"), convertCdn(endY, "y")), 6, (1, 97, 242), 2)

# 創建網絡圖
for way_id, data in way_dict.items():
    for i in range(len(data['node_refs']) - 1):
        u = data['node_refs'][i]
        v = data['node_refs'][i + 1]
        G.add_edge(u, v)

nowCdn = (convertCdn(121.32164781214851, "x"), convertCdn(24.99193515292301, "y"))
cv2.circle(img, nowCdn, 4, (252, 251, 223), -1)
cv2.circle(img, nowCdn, 6, (129, 91, 61), 2)
distance = cv2.norm(nowCdn, routeList[0])
print(distance)

# 顯示地圖
cv2.imshow('Map', cv2.flip(img, 0))
cv2.waitKey(0)
cv2.destroyAllWindows()
