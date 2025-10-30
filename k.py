import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import heapq
import random


class PerformanceAnalyzer:
    def __init__(self, graph_builder, search_algorithms):
        self.graph = graph_builder
        self.search = search_algorithms

    def run_performance_test(self, test_cases=10):
        """Гүйцэтгэлийн туршилт хийх"""
        results = {
            'BFS': {'time': [], 'memory': [], 'distance': [], 'steps': []},
            'DFS': {'time': [], 'memory': [], 'paths_found': []},
            'Dijkstra': {'time': [], 'memory': [], 'distance': []}
        }

        # Туршилтын цэгүүд
        nodes = list(self.graph.nodes.keys())

        for i in range(test_cases):
            if len(nodes) < 2:
                continue

            start, end = random.sample(nodes, 2)
            print(f"Туршилт {i + 1}/{test_cases}: {start} -> {end}")

            # BFS
            try:
                start_time = time.time()
                path, distance, metrics = self.search.bfs_shortest_path(start, end)
                bfs_time = time.time() - start_time

                if path:
                    results['BFS']['time'].append(bfs_time)
                    results['BFS']['memory'].append(metrics.get('memory', 0))
                    results['BFS']['distance'].append(distance)
                    results['BFS']['steps'].append(len(path) - 1)
            except Exception as e:
                print(f"BFS алдаа: {e}")

            # DFS
            try:
                start_time = time.time()
                paths, metrics = self.search.dfs_all_paths(start, end, max_paths=5)
                dfs_time = time.time() - start_time

                results['DFS']['time'].append(dfs_time)
                results['DFS']['memory'].append(metrics.get('memory', 0))
                results['DFS']['paths_found'].append(len(paths))
            except Exception as e:
                print(f"DFS алдаа: {e}")

            # Dijkstra
            try:
                start_time = time.time()
                path, distance, metrics = self.search.dijkstra_shortest_path(start, end)
                dijkstra_time = time.time() - start_time

                if path:
                    results['Dijkstra']['time'].append(dijkstra_time)
                    results['Dijkstra']['memory'].append(metrics.get('memory', 0))
                    results['Dijkstra']['distance'].append(distance)
            except Exception as e:
                print(f"Dijkstra алдаа: {e}")

        return results

    def plot_results(self, results):
        """Үр дүнг графикаар харуулах"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        algorithms = ['BFS', 'DFS', 'Dijkstra']
        colors = ['#4ecdc4', '#45b7d1', '#ff6b6b']

        # 1. Цаг хугацааны харьцуулалт
        time_data = []
        for algo in algorithms:
            if results[algo]['time']:
                time_data.append(np.mean(results[algo]['time']))
            else:
                time_data.append(0)

        bars1 = ax1.bar(algorithms, time_data, color=colors, alpha=0.8)
        ax1.set_title('Дундаж Гүйцэтгэлийн Хугацаа', fontsize=14, weight='bold')
        ax1.set_ylabel('Хугацаа (секунд)')
        ax1.bar_label(bars1, fmt='%.4f')
        ax1.grid(True, alpha=0.3)

        # 2. Санах ойн хэрэглээ
        memory_data = []
        for algo in algorithms:
            if results[algo]['memory']:
                memory_data.append(np.mean(results[algo]['memory']) / 1024)  # KB руу хөрвүүлэх
            else:
                memory_data.append(0)

        bars2 = ax2.bar(algorithms, memory_data, color=colors, alpha=0.8)
        ax2.set_title('Дундаж Санах Ойн Хэрэглээ', fontsize=14, weight='bold')
        ax2.set_ylabel('Санах ой (KB)')
        ax2.bar_label(bars2, fmt='%.1f')
        ax2.grid(True, alpha=0.3)

        # 3. Зайны харьцуулалт
        distance_data = []
        for algo in ['BFS', 'Dijkstra']:
            if results[algo]['distance']:
                distance_data.append(np.mean(results[algo]['distance']))
            else:
                distance_data.append(0)

        bars3 = ax3.bar(['BFS', 'Dijkstra'], distance_data,
                        color=[colors[0], colors[2]], alpha=0.8)
        ax3.set_title('Дундаж Зайны Харьцуулалт', fontsize=14, weight='bold')
        ax3.set_ylabel('Зай (метр)')
        ax3.bar_label(bars3, fmt='%.1f')
        ax3.grid(True, alpha=0.3)

        # 4. DFS-ийн олдсон замын тоо
        if results['DFS']['paths_found']:
            paths_found = np.mean(results['DFS']['paths_found'])
            bars4 = ax4.bar(['DFS'], [paths_found], color=colors[1], alpha=0.8)
            ax4.set_title('DFS: Олдсон Замын Дундаж Тоо', fontsize=14, weight='bold')
            ax4.set_ylabel('Замын тоо')
            ax4.bar_label(bars4, fmt='%.1f')
            ax4.grid(True, alpha=0.3)

        plt.suptitle('Алгоритмуудын Гүйцэтгэлийн Шинжилгээ', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        return self._print_statistics(results)

    def _print_statistics(self, results):
        """Статистик мэдээлэл хэвлэх"""
        print("\n" + "=" * 50)
        print("ГҮЙЦЭТГЭЛИЙН СТАТИСТИК")
        print("=" * 50)

        for algo in ['BFS', 'DFS', 'Dijkstra']:
            print(f"\n{algo} АЛГОРИТМ:")

            if results[algo]['time']:
                print(f"  • Хугацаа: {np.mean(results[algo]['time']):.6f} сек")
                print(f"  • Санах ой: {np.mean(results[algo]['memory']) / 1024:.1f} KB")

                if algo == 'BFS':
                    print(f"  • Дундаж алхам: {np.mean(results[algo]['steps']):.1f}")
                    print(f"  • Дундаж зай: {np.mean(results[algo]['distance']):.1f} м")
                elif algo == 'DFS':
                    print(f"  • Олдсон замын тоо: {np.mean(results[algo]['paths_found']):.1f}")
                elif algo == 'Dijkstra':
                    print(f"  • Дундаж зай: {np.mean(results[algo]['distance']):.1f} м")
            else:
                print(f"  • Туршилтын өгөгдөл байхгүй")


# Туршилт ажиллуулах
def run_performance_analysis():
    # Таны кодыг ашиглан
    from frontend import GraphBuilder, SearchAlgorithms

    # Граф үүсгэх
    graph_builder = GraphBuilder()
    graph_builder._create_test_data()  # Туршилтын өгөгдөл

    # Алгоритмууд
    search_algorithms = SearchAlgorithms(graph_builder)

    # Шинжилгээ
    analyzer = PerformanceAnalyzer(graph_builder, search_algorithms)
    results = analyzer.run_performance_test(test_cases=20)
    analyzer.plot_results(results)


if __name__ == "__main__":
    run_performance_analysis()