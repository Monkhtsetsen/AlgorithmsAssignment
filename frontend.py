import folium
import webbrowser
import os
from collections import deque
import heapq
import time


class GraphBuilder:
    def __init__(self):
        self.nodes = {}
        self.adjacency_list = {}

    def add_edge(self, node1: int, node2: int, weight: float):
        if node1 not in self.adjacency_list:
            self.adjacency_list[node1] = []
        self.adjacency_list[node1].append((node2, weight))


class SearchAlgorithms:
    def __init__(self, graph_builder):
        self.graph = graph_builder

    def calculate_path_distance(self, path):
        total = 0
        for i in range(len(path) - 1):
            for neighbor, weight in self.graph.adjacency_list.get(path[i], []):
                if neighbor == path[i + 1]:
                    total += weight
                    break
        return total

    def bfs_shortest_path(self, start: int, end: int):
        start_time = time.time()
        if start not in self.graph.adjacency_list or end not in self.graph.adjacency_list:
            return [], float('inf'), {'time': 0}

        visited = set()
        queue = deque([(start, [start])])
        visited.add(start)

        while queue:
            current, path = queue.popleft()
            if current == end:
                return path, self.calculate_path_distance(path), {'time': time.time() - start_time}

            for neighbor, weight in self.graph.adjacency_list.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return [], float('inf'), {'time': time.time() - start_time}

    def dfs_all_paths(self, start: int, end: int, max_paths: int = 5):
        start_time = time.time()
        all_paths = []

        def dfs(current, path, visited):
            if len(all_paths) >= max_paths:
                return
            if current == end:
                all_paths.append({
                    'path': path.copy(),
                    'distance': self.calculate_path_distance(path),
                    'steps': len(path) - 1
                })
                return
            visited.add(current)
            for neighbor, _ in self.graph.adjacency_list.get(current, []):
                if neighbor not in visited:
                    dfs(neighbor, path + [neighbor], visited.copy())

        dfs(start, [start], set())
        all_paths.sort(key=lambda x: x['distance'])
        return all_paths, {'time': time.time() - start_time}

    def dijkstra_shortest_path(self, start: int, end: int):
        start_time = time.time()
        if start not in self.graph.adjacency_list or end not in self.graph.adjacency_list:
            return [], float('inf'), {'time': 0}

        distances = {node: float('inf') for node in self.graph.nodes}
        predecessors = {node: None for node in self.graph.nodes}
        distances[start] = 0
        queue = [(0, start)]
        visited = set()

        while queue:
            current_distance, current_node = heapq.heappop(queue)
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
                    heapq.heappush(queue, (new_distance, neighbor))

        path = []
        current = end
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        if not path or path[0] != start:
            return [], float('inf'), {'time': time.time() - start_time}
        return path, distances[end], {'time': time.time() - start_time}


class MapVisualizer:
    def __init__(self, center_lat=47.8864, center_lon=106.9057):
        self.map = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')

    def add_path(self, coordinates, color='blue', weight=5, opacity=0.7, popup=None, offset_index=0):
        # PolyLine-ийг бага зэрэг шилжүүлж overlap багасгах
        offset = 0.00005 * offset_index
        offset_coords = [(lat + offset, lon + offset) for lat, lon in coordinates]
        folium.PolyLine(offset_coords, color=color, weight=weight, opacity=opacity, popup=popup).add_to(self.map)

    def add_marker(self, lat, lon, color='red', popup=None):
        folium.Marker([lat, lon], popup=popup, icon=folium.Icon(color=color)).add_to(self.map)

    def show_map(self, filename='map.html'):
        self.map.save(filename)
        webbrowser.open(f'file://{os.path.abspath(filename)}')

    def visualize_all_paths_clear(self, results, graph):
        path_styles = {
            'Dijkstra': {'color': '#FF0000', 'weight': 10, 'opacity': 1.0},
            'BFS': {'color': '#0000FF', 'weight': 8, 'opacity': 0.9},
            'DFS_shortest': {'color': '#FF00FF', 'weight': 6, 'opacity': 0.8},
            'DFS_other': {'color': '#00FF00', 'weight': 4, 'opacity': 0.6}
        }

        # Dijkstra
        if 'Dijkstra' in results and results['Dijkstra']['path']:
            coords = [graph.nodes[n] for n in results['Dijkstra']['path']]
            self.add_path(coords, **path_styles['Dijkstra'], popup=f"Dijkstra: {results['Dijkstra']['distance']:.0f}m", offset_index=0)

        # BFS
        if 'BFS' in results and results['BFS']['path']:
            coords = [graph.nodes[n] for n in results['BFS']['path']]
            self.add_path(coords, **path_styles['BFS'], popup=f"BFS: {results['BFS']['distance']:.0f}m", offset_index=1)

        # DFS
        if 'DFS' in results and 'paths' in results['DFS']:
            for i, path_data in enumerate(results['DFS']['paths']):
                coords = [graph.nodes[n] for n in path_data['path']]
                style = path_styles['DFS_shortest'] if i == 0 else path_styles['DFS_other']
                self.add_path(coords, **style, popup=f"DFS {i+1}: {path_data['distance']:.0f}m", offset_index=2+i)

        # Эхлэл, төгсгөлийн цэгүүд
        for algo, result in results.items():
            if 'path' in result and result['path']:
                start, end = result['path'][0], result['path'][-1]
                if start in graph.nodes:
                    self.add_marker(*graph.nodes[start], color='green', popup='Эхлэх')
                if end in graph.nodes:
                    self.add_marker(*graph.nodes[end], color='red', popup='Төгсгөл')
                break


def create_test_graph():
    graph = GraphBuilder()
    nodes = {
        0: (47.918, 106.917), 1: (47.920, 106.920), 2: (47.925, 106.925),
        3: (47.915, 106.930), 4: (47.910, 106.920), 5: (47.905, 106.915),
        6: (47.900, 106.910), 7: (47.895, 106.905)
    }
    edges = [
        (0, 1, 500), (1, 2, 800), (2, 3, 600),
        (0, 4, 700), (4, 5, 400), (5, 3, 300),
        (1, 4, 300), (2, 5, 500), (5, 6, 350),
        (6, 7, 450), (3, 7, 550), (0, 2, 1200)
    ]
    graph.nodes = nodes
    for n1, n2, w in edges:
        graph.add_edge(n1, n2, w)
        graph.add_edge(n2, n1, w)
    return graph


def interactive_system():
    graph = create_test_graph()
    search = SearchAlgorithms(graph)
    place_names = {
        0: "Сүхбаатар талбай", 1: "Улаанбаатар банк", 2: "ХУД", 3: "Найрамдал цэнгэлдэх",
        4: "Хан-Уул", 5: "13-р хороо", 6: "Сонгинохайрхан", 7: "Толгойт"
    }

    while True:
        print("\n📍 Боломжит цэгүүд:")
        for n, (lat, lon) in graph.nodes.items():
            print(f"{n}: {place_names.get(n, n)} ({lat:.3f}, {lon:.3f})")
        print("\n1. Зам төлөвлөх\n2. Гарах")
        choice = input("Сонголтоо оруулна уу: ")
        if choice == "2":
            break
        elif choice != "1":
            print("Буруу сонголт!")
            continue

        try:
            start_node = int(input("Эхлэх цэг: "))
            end_node = int(input("Төгсгөлийн цэг: "))
            if start_node not in graph.nodes or end_node not in graph.nodes:
                print("Буруу цэгийн дугаар!")
                continue

            results = {}
            path, dist, _ = search.dijkstra_shortest_path(start_node, end_node)
            results['Dijkstra'] = {'path': path, 'distance': dist, 'steps': len(path)-1}
            path, dist, _ = search.bfs_shortest_path(start_node, end_node)
            results['BFS'] = {'path': path, 'distance': dist, 'steps': len(path)-1}
            paths, _ = search.dfs_all_paths(start_node, end_node, 3)
            results['DFS'] = {'paths': paths}

            print("\n📊 Үр дүн:")
            for algo, result in results.items():
                if algo == 'DFS':
                    for i, p in enumerate(result['paths']):
                        print(f"DFS {i+1}: {p['distance']:.0f}m, {p['steps']} алхам")
                else:
                    print(f"{algo}: {result['distance']:.0f}m, {result['steps']} алхам")

            visualizer = MapVisualizer()
            visualizer.visualize_all_paths_clear(results, graph)
            visualizer.show_map('interactive_map.html')

        except Exception as e:
            print("Алдаа гарлаа:", e)


if __name__ == "__main__":
    interactive_system()
