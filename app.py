from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import geopandas as gpd
import math
from collections import deque, defaultdict
import heapq
import time
from typing import List, Tuple, Dict, Set, Optional

app = Flask(__name__)
CORS(app)

# Графын обьектууд
graph_builder = None
search_algorithms = None


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
        try:
            print("OSM өгөгдөл ачаалж байна...")
            gdf = gpd.read_file(shapefile_path)
            print(f"Амжилттай уншлаа: {len(gdf)} замын сегмент")

            node_id = 0
            road_segments = []

            for idx, road in gdf.iterrows():
                if hasattr(road.geometry, 'geom_type') and road.geometry.geom_type == 'LineString':
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

        except Exception as e:
            print(f"OSM файл уншихад алдаа гарлаа: {e}")
            print("Туршилтын өгөгдөл үүсгэж байна...")
            self._create_test_data()

    def _create_test_data(self):
        """Туршилтын өгөгдөл үүсгэх"""
        # Улаанбаатарын координатууд
        test_nodes = {
            0: (47.918, 106.917),  # Сүхбаатар талбай
            1: (47.920, 106.920),
            2: (47.925, 106.925),
            3: (47.915, 106.930),
            4: (47.910, 106.920),
            5: (47.905, 106.915),
            6: (47.900, 106.910),
            7: (47.895, 106.905),
        }

        test_edges = [
            (0, 1, 500), (1, 2, 800), (2, 3, 600),
            (0, 4, 700), (4, 5, 400), (5, 3, 300),
            (1, 4, 300), (2, 5, 500), (5, 6, 350),
            (6, 7, 450), (3, 7, 550)
        ]

        self.nodes = test_nodes
        for node1, node2, weight in test_edges:
            self.add_edge(node1, node2, weight)
            self.add_edge(node2, node1, weight)  # Хоёр чигийн зам

        print(f"Туршилтын өгөгдөл үүсгэв: {len(self.nodes)} цэг, {len(self.edges)} ирмэг")

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


class SearchAlgorithms:
    def __init__(self, graph_builder):
        self.graph = graph_builder

    def bfs_shortest_path(self, start: int, end: int) -> Tuple[List[int], float, Dict]:
        """BFS - Хамгийн цөөн алхам"""
        start_time = time.time()

        if start not in self.graph.adjacency_list or end not in self.graph.adjacency_list:
            return [], float('inf'), {'time': 0, 'memory': 0}

        visited = set()
        queue = deque([(start, [start])])
        visited.add(start)

        while queue:
            current, path = queue.popleft()

            if current == end:
                end_time = time.time()
                distance = sum(self.graph.edges.get((path[i], path[i + 1]), 0)
                               for i in range(len(path) - 1))
                return path, distance, {
                    'time': end_time - start_time,
                    'memory': len(visited) * 8,  # Ойролцоогоор санах ой
                    'nodes_visited': len(visited)
                }

            for neighbor, weight in self.graph.adjacency_list.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return [], float('inf'), {
            'time': time.time() - start_time,
            'memory': len(visited) * 8,
            'nodes_visited': len(visited)
        }

    def dfs_all_paths(self, start: int, end: int, max_paths: int = 10) -> Tuple[List[List[int]], Dict]:
        """DFS - Боломжит бүх зам"""
        start_time = time.time()
        paths = []
        visited_global = set()
        memory_usage = 0
        nodes_visited = 0

        def dfs(current, path, current_visited):
            nonlocal memory_usage, nodes_visited
            nodes_visited += 1

            if len(paths) >= max_paths:
                return

            if current == end:
                paths.append(path.copy())
                memory_usage = max(memory_usage, len(current_visited) * 8)
                return

            current_visited.add(current)
            visited_global.add(current)

            for neighbor, weight in self.graph.adjacency_list.get(current, []):
                if neighbor not in current_visited:
                    dfs(neighbor, path + [neighbor], current_visited.copy())

        dfs(start, [start], set())

        return paths, {
            'time': time.time() - start_time,
            'memory': memory_usage,
            'nodes_visited': nodes_visited
        }

    def dijkstra_shortest_path(self, start: int, end: int) -> Tuple[List[int], float, Dict]:
        """Dijkstra - Хамгийн богино зам"""
        start_time = time.time()

        if start not in self.graph.adjacency_list or end not in self.graph.adjacency_list:
            return [], float('inf'), {'time': 0, 'memory': 0}

        distances = {node: float('inf') for node in self.graph.nodes}
        predecessors = {node: None for node in self.graph.nodes}
        distances[start] = 0

        priority_queue = [(0, start)]
        visited = set()
        nodes_visited = 0

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            nodes_visited += 1

            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node == end:
                break

            for neighbor, weight in self.graph.adjacency_list.get(current_node, []):
                if neighbor in visited:
                    continue

                new_distance = current_distance + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))

        # Замыг бүтээх
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = predecessors[current]

        path.reverse()

        if not path or path[0] != start:
            return [], float('inf'), {
                'time': time.time() - start_time,
                'memory': len(visited) * 8,
                'nodes_visited': nodes_visited
            }

        return path, distances[end], {
            'time': time.time() - start_time,
            'memory': len(visited) * 8,
            'nodes_visited': nodes_visited
        }


@app.route('/api/path/shortest', methods=['POST'])
def shortest_path():
    """Хамгийн богино зам"""
    try:
        data = request.json
        start_lat = data['start_lat']
        start_lon = data['start_lon']
        end_lat = data['end_lat']
        end_lon = data['end_lon']

        start_node = graph_builder.find_nearest_node(start_lat, start_lon)
        end_node = graph_builder.find_nearest_node(end_lat, end_lon)

        path, distance, metrics = search_algorithms.dijkstra_shortest_path(start_node, end_node)

        coordinates = [graph_builder.nodes[node_id] for node_id in path]

        return jsonify({
            'success': True,
            'path': coordinates,
            'distance': distance,
            'metrics': metrics,
            'algorithm': 'Dijkstra',
            'start_node': start_node,
            'end_node': end_node
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/path/all', methods=['POST'])
def all_paths():
    """Бүх боломжит зам"""
    try:
        data = request.json
        start_lat = data['start_lat']
        start_lon = data['start_lon']
        end_lat = data['end_lat']
        end_lon = data['end_lon']
        max_paths = data.get('max_paths', 5)

        start_node = graph_builder.find_nearest_node(start_lat, start_lon)
        end_node = graph_builder.find_nearest_node(end_lat, end_lon)

        paths, metrics = search_algorithms.dfs_all_paths(start_node, end_node, max_paths)

        all_coordinates = []
        for path in paths:
            coordinates = [graph_builder.nodes[node_id] for node_id in path]
            distance = sum(graph_builder.edges.get((path[i], path[i + 1]), 0)
                           for i in range(len(path) - 1))
            all_coordinates.append({
                'path': coordinates,
                'distance': distance,
                'steps': len(path) - 1
            })

        return jsonify({
            'success': True,
            'paths': all_coordinates,
            'metrics': metrics,
            'algorithm': 'DFS',
            'start_node': start_node,
            'end_node': end_node
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/path/steps', methods=['POST'])
def minimal_steps():
    """Хамгийн цөөн алхам"""
    try:
        data = request.json
        start_lat = data['start_lat']
        start_lon = data['start_lon']
        end_lat = data['end_lat']
        end_lon = data['end_lon']

        start_node = graph_builder.find_nearest_node(start_lat, start_lon)
        end_node = graph_builder.find_nearest_node(end_lat, end_lon)

        path, distance, metrics = search_algorithms.bfs_shortest_path(start_node, end_node)

        coordinates = [graph_builder.nodes[node_id] for node_id in path]

        return jsonify({
            'success': True,
            'path': coordinates,
            'distance': distance,
            'steps': len(path) - 1,
            'metrics': metrics,
            'algorithm': 'BFS',
            'start_node': start_node,
            'end_node': end_node
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/nodes', methods=['GET'])
def get_nodes():
    """Бүх цэгүүдийг авах"""
    return jsonify({
        'success': True,
        'nodes': graph_builder.nodes,
        'total_nodes': len(graph_builder.nodes)
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'nodes': len(graph_builder.nodes),
        'edges': len(graph_builder.edges)
    })


@app.route('/api/test', methods=['GET'])
def test_route():
    """Туршилтын зам тооцоолол"""
    try:
        # Туршилтын цэгүүд
        start_node = 0
        end_node = 3

        # Бүх алгоритмаар тооцоолол хийх
        dijkstra_path, dijkstra_distance, dijkstra_metrics = search_algorithms.dijkstra_shortest_path(start_node,
                                                                                                      end_node)
        bfs_path, bfs_distance, bfs_metrics = search_algorithms.bfs_shortest_path(start_node, end_node)
        dfs_paths, dfs_metrics = search_algorithms.dfs_all_paths(start_node, end_node, 3)

        dfs_info = []
        for path in dfs_paths:
            distance = sum(graph_builder.edges.get((path[i], path[i + 1]), 0) for i in range(len(path) - 1))
            dfs_info.append({
                'path': [graph_builder.nodes[node_id] for node_id in path],
                'distance': distance,
                'steps': len(path) - 1
            })

        return jsonify({
            'success': True,
            'dijkstra': {
                'path': [graph_builder.nodes[node_id] for node_id in dijkstra_path],
                'distance': dijkstra_distance,
                'metrics': dijkstra_metrics
            },
            'bfs': {
                'path': [graph_builder.nodes[node_id] for node_id in bfs_path],
                'distance': bfs_distance,
                'metrics': bfs_metrics
            },
            'dfs': {
                'paths': dfs_info,
                'metrics': dfs_metrics
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


def initialize_api(shapefile_path: str = None):
    """API-г эхлүүлэх"""
    global graph_builder, search_algorithms

    graph_builder = GraphBuilder()

    if shapefile_path:
        graph_builder.load_osm_data(shapefile_path)
    else:
        graph_builder._create_test_data()

    search_algorithms = SearchAlgorithms(graph_builder)

    print("REST API бэлэн боллоо!")
    print(f"Нийт {len(graph_builder.nodes)} цэг, {len(graph_builder.edges)} ирмэг")
    return app


if __name__ == '__main__':
    # OSM файл байхгүй бол туршилтын өгөгдөл үүсгэнэ
    app = initialize_api('gis_osm_roads_free_1.shp')
    app.run(debug=True, port=5000, host='0.0.0.0')