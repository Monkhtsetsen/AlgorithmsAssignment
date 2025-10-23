import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import json


class PerformanceAnalyzer:
    def __init__(self, search_algorithms, graph_builder):
        self.search = search_algorithms
        self.graph = graph_builder

    def run_benchmarks(self, test_cases: List[Tuple[int, int]]) -> Dict:
        """Гүйцэтгэлийн шинжилгээ хийх"""
        results = {
            'Dijkstra': {'time': [], 'memory': [], 'distance': []},
            'BFS': {'time': [], 'memory': [], 'steps': []},
            'DFS': {'time': [], 'memory': [], 'paths_found': []}
        }

        for start, end in test_cases:
            # Dijkstra
            path, distance, metrics = self.search.dijkstra_shortest_path(start, end)
            if path:
                results['Dijkstra']['time'].append(metrics['time'])
                results['Dijkstra']['memory'].append(metrics['memory'])
                results['Dijkstra']['distance'].append(distance)

            # BFS
            path, distance, metrics = self.search.bfs_shortest_path(start, end)
            if path:
                results['BFS']['time'].append(metrics['time'])
                results['BFS']['memory'].append(metrics['memory'])
                results['BFS']['steps'].append(len(path) - 1)

            # DFS
            paths, metrics = self.search.dfs_all_paths(start, end, 5)
            results['DFS']['time'].append(metrics['time'])
            results['DFS']['memory'].append(metrics['memory'])
            results['DFS']['paths_found'].append(len(paths))

        return results

    def plot_results(self, results: Dict):
        """Үр дүнг графикаар харуулах"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Цаг хугацааны харьцуулалт
        algorithms = ['Dijkstra', 'BFS', 'DFS']
        times = [np.mean(results[algo]['time']) for algo in algorithms]

        ax1.bar(algorithms, times, color=['blue', 'green', 'red'])
        ax1.set_title('Дундаж гүйцэтгэлийн хугацаа')
        ax1.set_ylabel('Хугацаа (сек)')

        # Санах ойн хэрэглээ
        memory = [np.mean(results[algo]['memory']) for algo in algorithms]

        ax2.bar(algorithms, memory, color=['blue', 'green', 'red'])
        ax2.set_title('Дундаж санах ойн хэрэглээ')
        ax2.set_ylabel('Санах ой (байт)')

        # Зайны харьцуулалт
        if results['Dijkstra']['distance']:
            distances = [
                np.mean(results['Dijkstra']['distance']),
                np.mean([d for d in results['BFS']['distance'] if d != float('inf')]),
                np.mean([d for d in results['DFS']['distance'] if d != float('inf')])
            ]
            ax3.bar(algorithms, distances, color=['blue', 'green', 'red'])
            ax3.set_title('Дундаж зай')
            ax3.set_ylabel('Зай (метр)')

        # Олдсон замын тоо
        ax4.bar(['DFS'], [np.mean(results['DFS']['paths_found'])])
        ax4.set_title('DFS: Олдсон замын дундаж тоо')
        ax4.set_ylabel('Замын тоо')

        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def analyze_algorithms_theoretically():
    """Алгоритмуудын онолын шинжилгээ"""
    analysis = {
        'BFS': {
            'time_complexity': 'O(V + E)',
            'space_complexity': 'O(V)',
            'optimal': 'Алхамын хувьд оптимал',
            'use_case': 'Хамгийн цөөн алхам'
        },
        'DFS': {
            'time_complexity': 'O(V + E)',
            'space_complexity': 'O(V)',
            'optimal': 'Замын уртын хувьд оптимал биш',
            'use_case': 'Бүх боломжит замыг олох'
        },
        'Dijkstra': {
            'time_complexity': 'O((V + E) log V)',
            'space_complexity': 'O(V)',
            'optimal': 'Жинтэй графын хувьд оптимал',
            'use_case': 'Хамгийн богино зам'
        }
    }

    print("Алгоритмуудын онолын шинжилгээ:")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))

    return analysis