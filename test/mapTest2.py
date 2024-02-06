import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString
import math

# 本地 .osm 文件的路径
osm_file_path = "TYAIcampus2.osm"

# 读取 .osm 文件并创建图形数据
G = ox.graph_from_xml(osm_file_path)
fig, ax = ox.plot_graph(G, show=True, close=True, figsize=(10, 10), edge_color='#FFF', bgcolor='#F1F2F8', edge_linewidth=5.0, node_size=20, node_color="#FFC97E")


# 将图形数据转换为 GeoDataFrame
nodes, edges = ox.graph_to_gdfs(G)

# 计算角度的函数
def calculate_angle(geometry):
    coords = list(geometry.coords)
    angle = math.degrees(math.atan2(coords[-1][1] - coords[0][1], coords[-1][0] - coords[0][0]))
    return angle

# 创建一个 GeoDataFrame 存储节点连接的道路
node_roads = gpd.GeoDataFrame(columns=["node_id", "road_id", "angle"])

for idx, node in nodes.iterrows():
    # 获取节点连接的道路
    connected_edges = edges[(edges["u"] == idx) | (edges["v"] == idx)]

    for _, edge in connected_edges.iterrows():
        # 获取道路的几何信息
        geometry = edge["geometry"]

        # 计算角度并添加到 GeoDataFrame
        angle = calculate_angle(geometry)
        node_roads = node_roads.append({"node_id": idx, "road_id": edge["osmid"], "angle": angle}, ignore_index=True)

# 打印结果
print(node_roads.head())

# 打印结果
print(nodes.head())
print(edges.head())