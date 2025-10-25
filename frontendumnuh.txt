import folium
from folium import plugins
import webbrowser
import tempfile
import os
import math
from collections import deque, defaultdict
import heapq
import time
from typing import List, Tuple, Dict, Set, Optional


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
                    'memory': len(visited) * 8,
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


class MapVisualizer:
    def __init__(self, center_lat: float = 47.8864, center_lon: float = 106.9057):
        self.map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )

    def add_path(self, coordinates: list, color: str = 'blue', weight: int = 5,
                 opacity: float = 0.7, popup: str = None):
        """Зам нэмэх"""
        if len(coordinates) < 2:
            return

        folium.PolyLine(
            coordinates,
            color=color,
            weight=weight,
            opacity=opacity,
            popup=popup
        ).add_to(self.map)

    def add_marker(self, lat: float, lon: float, color: str = 'red',
                   popup: str = None):
        """Тэмдэглэгээ нэмэх"""
        folium.Marker(
            [lat, lon],
            popup=popup,
            icon=folium.Icon(color=color)
        ).add_to(self.map)

    def add_circle_marker(self, lat: float, lon: float, color: str = 'red',
                          radius: int = 5, popup: str = None):
        """Тойрог тэмдэглэгээ нэмэх"""
        folium.CircleMarker(
            [lat, lon],
            radius=radius,
            popup=popup,
            color=color,
            fill=True,
            fillColor=color
        ).add_to(self.map)

    def show_map(self, filename: str = 'map.html'):
        """Газрын зургийг харуулах"""
        self.map.save(filename)
        print(f"Газрын зураг {filename} файлд хадгалагдлаа")
        webbrowser.open(f'file://{os.path.abspath(filename)}')

    def visualize_algorithm_comparison(self, results: dict):
        """Алгоритмуудын харьцуулалт"""
        colors = {'BFS': 'red', 'DFS': 'green', 'Dijkstra': 'blue'}

        for algo, result in results.items():
            if result['path']:
                self.add_path(
                    result['path'],
                    color=colors.get(algo, 'gray'),
                    weight=6,
                    popup=f"{algo}: {result['distance']:.1f}m, {result.get('steps', '')} алхам"
                )

        # Эхлэл, төгсгөлийн цэгүүд
        if results:
            first_algo = list(results.keys())[0]
            if results[first_algo]['path']:
                start = results[first_algo]['path'][0]
                end = results[first_algo]['path'][-1]

                self.add_marker(start[0], start[1], 'green', 'Эхлэх цэг')
                self.add_marker(end[0], end[1], 'red', 'Төгсгөлийн цэг')

    def visualize_all_nodes(self, graph):
        """Бүх цэгүүдийг газрын зураг дээр харуулах"""
        for node_id, (lat, lon) in graph.nodes.items():
            self.add_circle_marker(
                lat, lon,
                color='blue',
                radius=3,
                popup=f'Цэг {node_id}'
            )


def test_system():
    """Системийг турших"""
    # Граф бүтээх
    graph = GraphBuilder()

    # Туршилтын өгөгдөл үүсгэх
    print("Туршилтын өгөгдөл үүсгэж байна...")

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

    graph.nodes = test_nodes
    for node1, node2, weight in test_edges:
        graph.add_edge(node1, node2, weight)
        graph.add_edge(node2, node1, weight)  # Хоёр чигийн зам

    # Хайлтын алгоритмууд
    search = SearchAlgorithms(graph)

    # Зам тооцоолох
    print("\nЗамын тооцоолол хийж байна...")

    # Бүх алгоритмаар тооцоолох
    results = {}

    print("Dijkstra алгоритмаар тооцоолж байна...")
    # Dijkstra
    path, distance, metrics = search.dijkstra_shortest_path(0, 7)
    results['Dijkstra'] = {
        'path': [graph.nodes[node] for node in path],
        'distance': distance,
        'metrics': metrics
    }
    print(f"✓ Dijkstra: {distance:.1f}m, {metrics['time']:.4f} сек, {metrics['nodes_visited']} цэг шалгасан")

    print("BFS алгоритмаар тооцоолж байна...")
    # BFS
    path, distance, metrics = search.bfs_shortest_path(0, 7)
    results['BFS'] = {
        'path': [graph.nodes[node] for node in path],
        'distance': distance,
        'steps': len(path) - 1,
        'metrics': metrics
    }
    print(
        f"✓ BFS: {distance:.1f}m, {len(path) - 1} алхам, {metrics['time']:.4f} сек, {metrics['nodes_visited']} цэг шалгасан")

    print("DFS алгоритмаар тооцоолж байна...")
    # DFS
    paths, metrics = search.dfs_all_paths(0, 7, 3)
    if paths:
        shortest_dfs_path = min(paths, key=lambda p: sum(
            graph.edges.get((p[i], p[i + 1]), 0) for i in range(len(p) - 1)
        ))
        results['DFS'] = {
            'path': [graph.nodes[node] for node in shortest_dfs_path],
            'distance': sum(graph.edges.get((shortest_dfs_path[i], shortest_dfs_path[i + 1]), 0)
                            for i in range(len(shortest_dfs_path) - 1)),
            'metrics': metrics
        }
        print(f"✓ DFS: {results['DFS']['distance']:.1f}m, {metrics['time']:.4f} сек, {len(paths)} зам олдсон")

    # Гүйцэтгэлийн шинжилгээ
    print("\n=== ГҮЙЦЭТГЭЛИЙН ШИНЖИЛГЭЭ ===")
    for algo, result in results.items():
        print(f"{algo}:")
        print(f"  - Зай: {result['distance']:.1f} метр")
        print(f"  - Цаг: {result['metrics']['time']:.4f} секунд")
        print(f"  - Санах ой: {result['metrics']['memory']} байт")
        print(f"  - Шалгасан цэгүүд: {result['metrics']['nodes_visited']}")

    # Газрын зураг дээр харуулах
    print("\nГазрын зураг үүсгэж байна...")
    visualizer = MapVisualizer()

    # Бүх цэгүүдийг харуулах
    visualizer.visualize_all_nodes(graph)

    # Алгоритмуудын замуудыг харуулах
    visualizer.visualize_algorithm_comparison(results)

    # Газрын зургийг харуулах
    visualizer.show_map('algorithm_comparison.html')

    print("\n✓ Газрын зураг амжилттай үүслээ! Браузер дээр харуулж байна...")


def interactive_test():
    """Интерактив туршилт"""
    # Граф бүтээх
    graph = GraphBuilder()

    # Өргөтгөсөн туршилтын өгөгдөл
    test_nodes = {
        0: (47.918, 106.917),  # Сүхбаатар талбай
        1: (47.920, 106.920),  # Улаанбаатар банк
        2: (47.925, 106.925),  # ХУД иргэний нисэхийн буудал
        3: (47.915, 106.930),  # Найрамдал цэнгэлдэх хүрээлэн
        4: (47.910, 106.920),  # Хан-Уул дүүрэг
        5: (47.905, 106.915),  # 13-р хороо
        6: (47.900, 106.910),  # Сонгинохайрхан дүүрэг
        7: (47.895, 106.905),  # Толгойт
        8: (47.930, 106.915),  # Бага тойруу
        9: (47.935, 106.910),  # Их тойруу
    }

    test_edges = [
        (0, 1, 500), (1, 2, 800), (2, 3, 600),
        (0, 4, 700), (4, 5, 400), (5, 3, 300),
        (1, 4, 300), (2, 5, 500), (5, 6, 350),
        (6, 7, 450), (3, 7, 550), (2, 8, 400),
        (8, 9, 300), (9, 1, 600)
    ]

    graph.nodes = test_nodes
    for node1, node2, weight in test_edges:
        graph.add_edge(node1, node2, weight)
        graph.add_edge(node2, node1, weight)

    search = SearchAlgorithms(graph)

    print("=== ИНТЕРАКТИВ ТУРШИЛТ ===")
    print("Боломжит цэгүүд:")
    for node_id, (lat, lon) in graph.nodes.items():
        print(f"Цэг {node_id}: ({lat:.3f}, {lon:.3f})")

    while True:
        try:
            print("\nШинэ зам тооцоолол (гарцах бол 'q' оруулна уу):")
            start_node = input("Эхлэх цэгийн дугаар: ")
            if start_node.lower() == 'q':
                break
            end_node = input("Төгсгөлийн цэгийн дугаар: ")
            if end_node.lower() == 'q':
                break

            start_node = int(start_node)
            end_node = int(end_node)

            if start_node not in graph.nodes or end_node not in graph.nodes:
                print("❌ Буруу цэгийн дугаар! Дахин оролдоно уу.")
                continue

            print(f"\nЦэг {start_node} -> Цэг {end_node} замыг тооцоолж байна...")

            # Бүх алгоритмаар тооцоолох
            results = {}

            # Dijkstra
            path, distance, metrics = search.dijkstra_shortest_path(start_node, end_node)
            results['Dijkstra'] = {
                'path': [graph.nodes[node] for node in path],
                'distance': distance,
                'metrics': metrics
            }

            # BFS
            path, distance, metrics = search.bfs_shortest_path(start_node, end_node)
            results['BFS'] = {
                'path': [graph.nodes[node] for node in path],
                'distance': distance,
                'steps': len(path) - 1,
                'metrics': metrics
            }

            # DFS
            paths, metrics = search.dfs_all_paths(start_node, end_node, 2)
            if paths:
                shortest_dfs_path = min(paths, key=lambda p: sum(
                    graph.edges.get((p[i], p[i + 1]), 0) for i in range(len(p) - 1)
                ))
                results['DFS'] = {
                    'path': [graph.nodes[node] for node in shortest_dfs_path],
                    'distance': sum(graph.edges.get((shortest_dfs_path[i], shortest_dfs_path[i + 1]), 0)
                                    for i in range(len(shortest_dfs_path) - 1)),
                    'metrics': metrics
                }

            # Үр дүнг хэвлэх
            print("\n=== ҮР ДҮН ===")
            for algo, result in results.items():
                if result['path']:
                    print(f"{algo}:")
                    print(f"  - Зай: {result['distance']:.1f} метр")
                    print(f"  - Алхам: {len(result['path']) - 1}")
                    print(f"  - Цаг: {result['metrics']['time']:.4f} сек")
                else:
                    print(f"{algo}: Зам олдсонгүй")

            # Газрын зураг үүсгэх
            visualizer = MapVisualizer()
            visualizer.visualize_all_nodes(graph)
            visualizer.visualize_algorithm_comparison(results)

            filename = f'path_{start_node}_to_{end_node}.html'
            visualizer.show_map(filename)

            print(f"\n✓ Газрын зураг {filename} файлд хадгалагдлаа!")

        except ValueError:
            print("❌ Тоо оруулна уу!")
        except Exception as e:
            print(f"❌ Алдаа: {e}")


if __name__ == "__main__":
    print("Улаанбаатар хотын зам тооцоолуур")
    print("1. Туршилтын систем")
    print("2. Интерактив туршилт")

    choice = input("Сонголтоо оруулна уу (1 эсвэл 2): ")

    if choice == "1":
        test_system()
    elif choice == "2":
        interactive_test()
    else:
        print("Буруу сонголт! Туршилтын систем ажиллаж байна...")
        test_system()