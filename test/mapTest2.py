from curses import raw
from turtle import st
import xml.etree.ElementTree as ET
# from aem import con
import osmnx as ox
import networkx as nx
import cv2
import numpy as np
import requests
import time
import math
from rich import traceback, print, console
from rich.table import Table
from rich.live import Live

traceback.install(show_locals=True)
console = console.Console()

def convertCdn(num, type):
    if type == "x":
        cdnX = int((num - min_x) * scale_x)
        # cdnX = width - cdnX
        return cdnX
    elif type == "y":
        cdnY = int((num - min_y) * scale_y)
        cdnY = height - cdnY
        return cdnY

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




# print(root.findall("way"))

# 獲取路徑的範圍
min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf

# 遍歷所有way並僅顯示service等於driveway的路徑
roadSide = {}
roadMapping = {}
for way_id, data in way_dict.items():
    if 'service' in data['tags'] and data['tags']['service'] == 'driveway':
        try:
            # print("target:", data['tags']['target'], "::>", way_id)
            roadSide[way_id] = data['tags']['target']
        except:
            pass
        route = []
        # 從way中提取節點座標
        roadPoint = []
        for node_id in data['node_refs']:
            if node_id in node_dict:
                route.append(node_dict[node_id])
            roadPoint.append(node_id)
        route = np.array(route)
        roadMapping[way_id] = roadPoint

        # 更新範圍
        min_x = min(min_x, np.min(route[:, 1]))
        max_x = max(max_x, np.max(route[:, 1]))
        min_y = min(min_y, np.min(route[:, 0]))
        max_y = max(max_y, np.max(route[:, 0]))

# 計算縮放比例
width = 1000
height = 760
scale_x = width / (max_x - min_x)
scale_y = height / (max_y - min_y)

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
            cv2.line(img, (ux, uy), (vx, vy), (245, 232, 217), 20)
            try:
                site = data['tags']['target']
                canGoWay.append([(ux, uy), (vx, vy), site])
            except:
                pass
        
        # 繪製節點
        for node in route:
            x = int((node[1] - min_x) * scale_x - 1)
            y = height - int((node[0] - min_y) * scale_y - 1)
            cv2.circle(img, (x, y), 5, (212, 171, 134), -1)

originImg = img.copy()

###########
# 規劃路線 #
###########
G = ox.graph_from_point((24.99250, 121.32032), dist=200, network_type='drive_service')

# 北24.99471° 東121.32081°
startY, startX = 24.994125, 121.319876
endY, endX = 24.992127, 121.320612

origin = ox.distance.nearest_nodes(G, startX, startY)
destination = ox.distance.nearest_nodes(G, endX, endY)

route = nx.shortest_path(G, origin, destination)
lastNode = (convertCdn(startX, "x"), convertCdn(startY, "y"))
routeList = []
rawRoute = [[startX, startY]]
roadAngle = []
for node in route:
    x, y = G.nodes[node]['x'], G.nodes[node]['y']
    rawRoute.append([x, y])
    x = convertCdn(x, "x")
    y = convertCdn(y, "y")
    cv2.line(img, lastNode, (x, y), (105, 66, 48), 10)
    #計算結點間角度並儲存
    lineAngle = np.arctan2(y - lastNode[1], x - lastNode[0])
    lineAngle = int(np.degrees(lineAngle))
    lineAngle += 90
    lineAngle += 360 if lineAngle < 0 else 0
    roadAngle.append(lineAngle)
    cv2.putText(img, str(lineAngle), (int((x+lastNode[0])/2), int((y+lastNode[1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 2)
    routeList.append((x, y))
    print(f"[white]{lastNode}")
    lastNode = (x, y)
    # cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
rawRoute.append([endX, endY])
print(f"{rawRoute}[green]")
cv2.line(img, lastNode, (convertCdn(endX, "x"), convertCdn(endY, "y")), (105, 66, 48), 10)
cv2.circle(img, (convertCdn(startX, "x"), convertCdn(startY, "y")), 15, (1, 97, 242), -1)
cv2.circle(img, (convertCdn(endX, "x"), convertCdn(endY, "y")), 15, (255, 255, 255), -1)
cv2.circle(img, (convertCdn(endX, "x"), convertCdn(endY, "y")), 15, (1, 97, 242), 2)

def refreshRoute(firstRoute ,angleList):
    x, y = G.nodes[firstRoute[0]]['x'], G.nodes[firstRoute[0]]['y']
    endX, endY = G.nodes[firstRoute[-1]]['x'], G.nodes[firstRoute[-1]]['y']
    lastNode = (convertCdn(x, "x"), convertCdn(y, "y"))
    for i, node in enumerate(firstRoute):
        x, y = G.nodes[node]['x'], G.nodes[node]['y']
        x = convertCdn(x, "x")
        y = convertCdn(y, "y")
        print(lastNode)
        cv2.line(img, lastNode, (x, y), (105, 66, 48), 10)
        cv2.putText(img, str(angleList[i]), (int((x+lastNode[0])/2), int((y+lastNode[1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 2)
        lastNode = (x, y)
    rawRoute.append([endX, endY])
    print(f"[green]{rawRoute}")
    cv2.line(img, lastNode, (convertCdn(endX, "x"), convertCdn(endY, "y")), (105, 66, 48), 10)
    # cv2.circle(img, (convertCdn(startX, "x"), convertCdn(startY, "y")), 15, (1, 97, 242), -1)
    cv2.circle(img, (convertCdn(endX, "x"), convertCdn(endY, "y")), 15, (255, 255, 255), -1)
    cv2.circle(img, (convertCdn(endX, "x"), convertCdn(endY, "y")), 15, (1, 97, 242), 2)

# 創建網絡圖
for way_id, data in way_dict.items():
    for i in range(len(data['node_refs']) - 1):
        u = data['node_refs'][i]
        v = data['node_refs'][i + 1]
        G.add_edge(u, v)

# distance = cv2.norm(nowCdn, routeList[0])
# print(distance)

def calculate_distance(point1, point2):

  x1, y1 = point1
  x2, y2 = point2
  distance = math.hypot(x2 - x1, y2 - y1)
  return distance

# testDist = calculate_distance((0, 0), (3, 4))
# print(f"[light_coral]{testDist}")

def generate_table(Lat, Lon, Heading, pointDist, pointAngle, state) -> Table:
    """Make a new table."""
    table = Table()
    table.add_column("經度")
    table.add_column("緯度")
    table.add_column("方向")
    table.add_column("最近點距離")
    table.add_column("最近點相差角")
    table.add_column("狀態")

    table.add_row(
        f"{Lat:3.4f}", f"{Lon:3.4f}", f"{Heading}", f"{pointDist:3.2f}", f"{pointAngle}", "[green]Connecting" if state else "[red]Disconnected"
    )
        
    return table

netState = False
copyMap = img.copy()
nearPoint = True
with Live(generate_table(0, 0, 0, 0, 0, 0)) as live:
    while len(routeList) >= 0:

        img = copyMap.copy()
        stateCode = 400

        # 發送GET請求
        try:
            response = requests.get('http://127.0.0.1:8600/getData')
            stateCode = response.status_code
        except:
            stateCode = 400

        # 檢查請求是否成功
        if stateCode == 200:
            netState = True
            # 將回應的內容解析為JSON
            data = response.json()
            # console.print(data)
            nowCdn = (convertCdn(float(data["latitude"]), "x"), convertCdn(float(data["longitude"]), "y"))
            # print('3111197520' in roadMapping)
            # for k in list(roadMapping.keys()):
            #     if str(route[0]) in roadMapping[k]:
            #         print("[pink]You are in", roadSide[k])
            #         break
            # 北24.99247° 東121.32210°
            # 24.994203, 121.320079
            # nowCdn = (convertCdn(121.320079, "x"), convertCdn(24.994203, "y"))
            rawHeading = int(data["heading"])
            heading = rawHeading - 90
            angleDist = rawHeading-roadAngle[1]
            angleDist = angleDist if abs(angleDist) < 180 else angleDist + 360
            angleDist = abs(angleDist)
            pointDist = calculate_distance(nowCdn, routeList[0])
            live.update(generate_table(float(data["latitude"]), float(data["longitude"]), rawHeading, pointDist, angleDist, netState))


            if pointDist < 30 and angleDist < 20 and nearPoint == True:
                nearPoint = False
                routeList.pop(0)
                roadAngle.pop(0)
                route.pop(0)
                print("past a Point")
                print(roadAngle)
                print("routeList:", routeList)
                if len(routeList) == 0:
                    break
                elif len(routeList) >= 2:
                    img = originImg.copy()
                    refreshRoute(route, roadAngle)
                    copyMap = img.copy()
            elif pointDist > 30 and nearPoint == False:
                nearPoint = True
                

            # 線的長度
            length = 100

            # 計算終點座標
            end_x = int(nowCdn[0] + length * math.cos(math.radians(heading)))
            end_y = int(nowCdn[1] + length * math.sin(math.radians(heading)))
            end_point = (end_x, end_y)

            # 畫線
            cv2.line(img, nowCdn, end_point, (255, 0, 0), 2)

            cv2.circle(img, nowCdn, 10, (252, 251, 223), -1)
            cv2.circle(img, nowCdn, 12, (129, 91, 61), 2)
        else:
            netState = False
            print('Failed to get data:',stateCode)
            live.update(generate_table(-1, -1, -1, -1, -1, netState))
        # time.sleep(0.1)
        cv2.imshow('Map', img)
        if cv2.waitKey(100) == 27:
            break

# 顯示地圖
cv2.destroyAllWindows()
