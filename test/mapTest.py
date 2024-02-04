import osmnx as ox
import networkx as nx

G = ox.graph_from_point((24.99344,121.32125), dist=500, network_type='drive_service')

ox.plot_graph(G)