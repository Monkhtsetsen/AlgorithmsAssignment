from collections import deque, defaultdict
import heapq
from typing import List, Tuple, Dict, Set
import time


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
                    'memory': len(visited) * 8  # Ойролцоогоор санах ой
                }

            for neighbor, weight in self.graph.adjacency_list.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return [], float('inf'), {'time': time.time() - start_time, 'memory': len(visited) * 8}

    def dfs_all_paths(self, start: int, end: int, max_paths: int = 10) -> Tuple[List[List[int]], Dict]:
        """DFS - Боломжит бүх зам"""
        start_time = time.time()
        paths = []
        visited = set()
        memory_usage = 0

        def dfs(current, path, current_visited):
            nonlocal memory_usage
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
            'memory': memory_usage
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

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

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

        if path[0] != start:
            return [], float('inf'), {'time': time.time() - start_time, 'memory': len(visited) * 8}

        return path, distances[end], {
            'time': time.time() - start_time,
            'memory': len(visited) * 8
        }