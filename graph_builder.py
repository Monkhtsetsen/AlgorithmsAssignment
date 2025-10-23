import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import math


class GraphBuilder:
    def __init__(self):
        self.nodes = {}  # node_id -> (lat, lon)
        self.edges = {}  # (node1, node2) -> weight
        self.adjacency_list = {}  # node_id -> [(neighbor, weight)]

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Хоёр цэгийн хоорондох зайг тооцоолох"""
        R = 6371000  # Дэлхийн радиус метрээр

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) * math.sin(delta_lat / 2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) * math.sin(delta_lon / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def load_osm_data(self, shapefile_path: str):
        """OSM өгөгдлийг ачааллах"""
        print("OSM өгөгдөл ачаалж байна...")
        gdf = gpd.read_file(shapefile_path)

        node_id = 0
        road_segments = []

        for idx, road in gdf.iterrows():
            if road.geometry.geom_type == 'LineString':
                coords = list(road.geometry.coords)

                # Замын цэгүүдийг нэмэх
                for i, (lon, lat) in enumerate(coords):
                    self.nodes[node_id] = (lat, lon)

                    if i > 0:
                        # Замын сегмент үүсгэх
                        prev_lat, prev_lon = self.nodes[node_id - 1]
                        distance = self.haversine_distance(prev_lat, prev_lon, lat, lon)

                        # Замын жинг тодорхойлох
                        weight = distance

                        # Нэг чигийн зам эсэхийг шалгах
                        if hasattr(road, 'oneway') and road.oneway == 'yes':
                            self.add_edge(node_id - 1, node_id, weight)
                        else:
                            self.add_edge(node_id - 1, node_id, weight)
                            self.add_edge(node_id, node_id - 1, weight)

                    node_id += 1

        print(f"Амжилттай ачааллаа: {len(self.nodes)} цэг, {len(self.edges)} ирмэг")

    def add_edge(self, node1: int, node2: int, weight: float):
        """Ирмэг нэмэх"""
        self.edges[(node1, node2)] = weight

        if node1 not in self.adjacency_list:
            self.adjacency_list[node1] = []
        self.adjacency_list[node1].append((node2, weight))

    def find_nearest_node(self, lat: float, lon: float) -> int:
        """Өгөгдсөн координаттай хамгийн ойрхон цэгийг олох"""
        min_distance = float('inf')
        nearest_node = None

        for node_id, (node_lat, node_lon) in self.nodes.items():
            distance = self.haversine_distance(lat, lon, node_lat, node_lon)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id

        return nearest_node