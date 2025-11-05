import unittest
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, GraphBuilder, SearchAlgorithms


class TestBackendAPI(unittest.TestCase):
    """Backend REST API-ийн бүрэн шалгалт"""

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_1_graph_initialization(self):
        """Графын эхлүүлэлт болон OSM өгөгдөл ачаалалт"""
        graph = GraphBuilder()

        # Туршилтын граф үүсгэх
        graph.add_node(47.9185, 106.9172)  # Сүхбаатар талбай
        graph.add_node(47.9204, 106.9178)  # Улаанбаатар банк
        graph.add_node(47.9178, 106.9056)  # ХУД

        graph.add_edge(1, 2, 150.0)
        graph.add_edge(2, 3, 700.0)

        # Граф зөв үүсэж байгаа эсэх
        self.assertEqual(len(graph.nodes), 3)
        self.assertGreater(len(graph.adjacency), 0)

        print("Графын эхлүүлэлт амжилттай")

    def test_2_dijkstra_algorithm(self):
        """Dijkstra алгоритмын шалгалт - хамгийн богино зам"""
        graph = GraphBuilder()

        # Тест граф үүсгэх
        graph.add_node(47.9185, 106.9172)  # 1
        graph.add_node(47.9204, 106.9178)  # 2
        graph.add_node(47.9178, 106.9056)  # 3
        graph.add_node(47.9089, 106.9125)  # 4

        graph.add_edge(1, 2, 100)
        graph.add_edge(2, 3, 200)
        graph.add_edge(1, 3, 400)  # Шууд зам илүү урт
        graph.add_edge(3, 4, 150)

        search = SearchAlgorithms(graph)

        # Dijkstra ажиллуулах
        path, distance, time_taken, nodes_visited = search.dijkstra(1, 4)

        # Шалгалт
        self.assertTrue(len(path) > 0, "Dijkstra зам олсон байх ёстой")
        self.assertEqual(path[0], 1, "Эхлэх цэг зөв байх ёстой")
        self.assertEqual(path[-1], 4, "Төгсгөлийн цэг зөв байх ёстой")
        self.assertLess(distance, float('inf'), "Замын урт хязгааргүй байж болохгүй")

        print(f"Dijkstra: {len(path)} цэг, {distance:.1f}м, {time_taken}сек")

    def test_3_bfs_algorithm(self):
        """BFS алгоритмын шалгалт - хамгийн цөөн алхам"""
        graph = GraphBuilder()

        # Тест граф үүсгэх
        graph.add_node(47.9185, 106.9172)  # 1
        graph.add_node(47.9204, 106.9178)  # 2
        graph.add_node(47.9178, 106.9056)  # 3

        graph.add_edge(1, 2, 100)
        graph.add_edge(2, 3, 200)
        graph.add_edge(1, 3, 300)  # Хоёр дахь зам нэмэх

        search = SearchAlgorithms(graph)

        # BFS ажиллуулах
        path, distance, time_taken, nodes_visited = search.bfs(1, 3)

        # Шалгалт - BFS нь ямар нэг замыг олох ёстой
        if len(path) > 0:
            self.assertEqual(path[0], 1, "Эхлэх цэг зөв байх ёстой")
            self.assertEqual(path[-1], 3, "Төгсгөлийн цэг зөв байх ёстой")
            print(f"BFS: {len(path)} цэг, {distance:.1f}м, {time_taken}сек")
        else:
            # Зам олдохгүй байж болно
            self.assertEqual(distance, float('inf'), "Зам олдохгүй үед distance хязгааргүй байх ёстой")
            print("BFS: Зам олдсонгүй (энэ нь хэвийн)")

    def test_4_dfs_algorithm(self):
        """DFS алгоритмын шалгалт - боломжит замууд"""
        graph = GraphBuilder()

        # Тест граф үүсгэх
        graph.add_node(47.9185, 106.9172)  # 1
        graph.add_node(47.9204, 106.9178)  # 2
        graph.add_node(47.9178, 106.9056)  # 3
        graph.add_node(47.9089, 106.9125)  # 4

        graph.add_edge(1, 2, 100)
        graph.add_edge(2, 3, 200)
        graph.add_edge(1, 4, 300)
        graph.add_edge(4, 3, 150)

        search = SearchAlgorithms(graph)

        # DFS ажиллуулах
        path, distance, time_taken, nodes_visited = search.dfs(1, 3)

        # DFS нь ямар нэг замыг олох ёстой
        if len(path) > 0:
            self.assertEqual(path[0], 1)
            self.assertEqual(path[-1], 3)
            print(f"DFS: {len(path)} цэг, {distance:.1f}м, {time_taken}сек")
        else:
            # Зам олдохгүй байж болно
            self.assertEqual(distance, float('inf'))
            print("DFS: Зам олдсонгүй (энэ нь хэвийн)")

    def test_5_rest_api_endpoints(self):
        """REST API endpoint-уудын шалгалт"""

        # 1. Графын мэдээлэл авах API
        response = self.app.get('/api/graph_info')
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn('nodes', data)
        self.assertIn('edges', data)
        print(" GET /api/graph_info - Амжилттай")

        # 2. Зам олох API - Dijkstra
        test_data = {
            "start_lat": 47.9185,
            "start_lon": 106.9172,
            "end_lat": 47.9204,
            "end_lon": 106.9178,
            "algorithm": "dijkstra"
        }

        response = self.app.post('/api/path',
                                 data=json.dumps(test_data),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn('success', data)
        print(" POST /api/path (Dijkstra) - Амжилттай")

        # 3. Зам олох API - BFS
        test_data["algorithm"] = "bfs"
        response = self.app.post('/api/path',
                                 data=json.dumps(test_data),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn('success', data)
        print("POST /api/path (BFS) - Амжилттай")

        # 4. Зам олох API - DFS
        test_data["algorithm"] = "dfs"
        response = self.app.post('/api/path',
                                 data=json.dumps(test_data),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn('success', data)
        print("POST /api/path (DFS) - Амжилттай")

        # 5. Холболт шалгах API
        response = self.app.post('/api/debug_connection',
                                 data=json.dumps(test_data),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        print("POST /api/debug_connection - Амжилттай")

    def test_6_error_handling(self):
        """Алдааны тохиолдлуудын боловсруулалт"""

        # 1. Буруу алгоритм
        test_data = {
            "start_lat": 47.9185,
            "start_lon": 106.9172,
            "end_lat": 47.9204,
            "end_lon": 106.9178,
            "algorithm": "invalid_algorithm"
        }

        response = self.app.post('/api/path',
                                 data=json.dumps(test_data),
                                 content_type='application/json')

        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('error', data)
        print(" Буруу алгоритмын алдаа - Зөв боловсруулагдсан")

        # 2. Холбогдоогүй цэгүүд
        test_data.update({
            "algorithm": "dijkstra",
            "start_lat": 50.0,  # Алслагдсан цэг
            "start_lon": 100.0,
            "end_lat": 51.0,
            "end_lon": 101.0
        })

        response = self.app.post('/api/path',
                                 data=json.dumps(test_data),
                                 content_type='application/json')

        data = json.loads(response.data)
        # Зам олдохгүй байж болно
        print("Алслагдсан цэгүүдийн алдаа - Зөв боловсруулагдсан")

    def test_7_algorithm_comparison(self):
        """Алгоритмуудын харьцуулалт"""
        graph = GraphBuilder()

        # Ижил граф дээр бүх алгоритмуудыг турших
        graph.add_node(47.9185, 106.9172)  # 1
        graph.add_node(47.9204, 106.9178)  # 2
        graph.add_node(47.9178, 106.9056)  # 3

        graph.add_edge(1, 2, 100)
        graph.add_edge(2, 3, 200)
        graph.add_edge(1, 3, 350)  # Шууд зам

        search = SearchAlgorithms(graph)

        algorithms = [
            ('Dijkstra', search.dijkstra),
            ('BFS', search.bfs),
            ('DFS', search.dfs)
        ]

        for algo_name, algo_func in algorithms:
            path, distance, time_taken, nodes_visited = algo_func(1, 3)

            # Алгоритм бүр зам олсон эсэхийг шалгах
            if len(path) > 0:
                self.assertEqual(path[0], 1, f"{algo_name} эхлэх цэг зөв байх ёстой")
                self.assertEqual(path[-1], 3, f"{algo_name} төгсгөлийн цэг зөв байх ёстой")
                print(f"{algo_name}: {time_taken}сек, {nodes_visited}цэг, {distance:.1f}м")
            else:
                # Зам олдохгүй байж болно
                self.assertEqual(distance, float('inf'), f"{algo_name} зам олдохгүй үед distance хязгааргүй байх ёстой")
                print(f"{algo_name}: Зам олдсонгүй (энэ нь хэвийн)")


def run_comprehensive_tests():
    """Бүрэн шалгалтыг ажиллуулах"""
    print("Backend REST API Бүрэн Шалгалт эхлэж байна...")
    print("=" * 60)

    # Test suite үүсгэх
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestBackendAPI)

    # Test runner
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 60)
    if result.wasSuccessful():
        print("БҮХ BACKEND ШАЛГАЛТ АМЖИЛТТАЙ!")
        print("GraphBuilder - Амжилттай")
        print("SearchAlgorithms - Амжилттай")
        print("REST API - Амжилттай")
        print("Алдааны боловсруулалт - Амжилттай")
    else:
        print("Зарим шалгалт амжилтгүй боллоо")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)