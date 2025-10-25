import unittest
import matplotlib.pyplot as plt
import networkx as nx

# -------------------------------
# Graph class
# -------------------------------
class GraphBuilder:
    def __init__(self):
        self.adj = {}

    def add_edge(self, u, v, w=1):
        if u not in self.adj:
            self.adj[u] = {}
        self.adj[u][v] = w

    def dfs_all_paths(self, start, end, path=None):
        if path is None:
            path = [start]
        if start == end:
            return [path]
        paths = []
        for node in self.adj.get(start, {}):
            if node not in path:
                newpaths = self.dfs_all_paths(node, end, path + [node])
                paths.extend(newpaths)
        return paths

    def bfs_shortest_path(self, start, end):
        from collections import deque
        queue = deque([(start, [start], 0)])
        visited = set()
        while queue:
            node, path, dist = queue.popleft()
            if node == end:
                return path, dist
            visited.add(node)
            for neighbor, w in self.adj.get(node, {}).items():
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor], dist + w))
        return None, float('inf')

    def dijkstra(self, start, end):
        import heapq
        heap = [(0, start, [start])]
        visited = set()
        while heap:
            dist, node, path = heapq.heappop(heap)
            if node == end:
                return path, dist
            if node in visited:
                continue
            visited.add(node)
            for neighbor, w in self.adj.get(node, {}).items():
                if neighbor not in visited:
                    heapq.heappush(heap, (dist + w, neighbor, path + [neighbor]))
        return None, float('inf')

# -------------------------------
# Graph visualizer
# -------------------------------
class MapVisualizer:
    @staticmethod
    def draw_graph(graph, path=None, filename=None):
        G = nx.DiGraph()
        for node, edges in graph.adj.items():
            for neighbor, weight in edges.items():
                G.add_edge(node, neighbor, weight=weight)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=12)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        if path:
            edge_list = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color='red', width=3)

        if filename:
            plt.savefig(filename)  # PNG-д хадгалах
        plt.close()  # popup window гаргахгүй

# -------------------------------
# Unit tests
# -------------------------------
class TestGraphAlgorithms(unittest.TestCase):
    def setUp(self):
        self.graph = GraphBuilder()
        self.graph.add_edge("A", "B", 1)
        self.graph.add_edge("B", "C", 2)
        self.graph.add_edge("A", "C", 4)
        self.graph.add_edge("C", "D", 1)
        self.graph.add_edge("B", "D", 5)
        self.graph.add_edge("D", "E", 3)
        self.graph.add_edge("C", "E", 2)

    def test_dfs_all_paths(self):
        paths = self.graph.dfs_all_paths("A", "E")
        min_len = min(len(p) for p in paths)
        self.assertEqual(min_len, 3)
        # Хадгалах
        MapVisualizer.draw_graph(self.graph, paths[0], filename="dfs_path.png")

    def test_bfs_shortest_path(self):
        path, distance = self.graph.bfs_shortest_path("A", "E")
        self.assertEqual(path, ["A", "C", "E"])
        self.assertEqual(distance, 6)
        MapVisualizer.draw_graph(self.graph, path, filename="bfs_path.png")

    def test_dijkstra_shortest_path(self):
        path, distance = self.graph.dijkstra("A", "E")
        self.assertEqual(path, ["A", "B", "C", "E"])
        self.assertEqual(distance, 5)
        MapVisualizer.draw_graph(self.graph, path, filename="dijkstra_path.png")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    unittest.main()
