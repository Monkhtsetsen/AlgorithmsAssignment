from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import geopandas as gpd
from shapely.geometry import LineString
import math, heapq, time
from collections import deque#bfs double ended queue

app = Flask(__name__)
CORS(app)

class GraphBuilder:
    def __init__(self):
        self.nodes = {}
        self.adjacency = {} #neg node oos nogoo node ruu holbogdson zam
        self.node_count = 0
#hoyr tsegiin hoorondoh zaig metreer oldog
    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371000 #delhiin radius
        phi1, phi2 = math.radians(lat1), math.radians(lat2) #urgurguudiig radiand huvirgaj bna
        #delhiin gadarguu deerh hoyr tsegiin murui zaig tootsoh tomyo
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#neg id uusgeed coordinatiig hadgalna
    def add_node(self, lat, lon):
        self.node_count += 1
        self.nodes[self.node_count] = (lat, lon)
        return self.node_count

    def add_edge(self, n1, n2, weight):
        self.adjacency.setdefault(n1, []).append((n2, weight))

    def load_osm_data(self, shapefile_path):
        print("Loading OSM shapefile:", shapefile_path)
        gdf = gpd.read_file(shapefile_path)
        print("Shapefile columns:", list(gdf.columns))

        # Бүх төрлийн замыг оруулах
        road_filters = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary',
                        'residential', 'unclassified', 'service', 'living_street']
#fcall buyu zamiin turul baihgui esehiig shalgaj bna
        if 'fclass' in gdf.columns:
            filters = gdf['fclass'].isin(road_filters)
            drivable = gdf[filters].reset_index(drop=True)
        else:
            # fclass байхгүй бол бүх замыг авах
            drivable = gdf
            print("!!!'fclass' column not found, using all roads")

        coord_to_node = {}
#davhardsan coordinatad shine node uusgehgui baih
        def get_node(lat, lon):
            key = (round(lat, 6), round(lon, 6))
            if key not in coord_to_node:
                node_id = self.add_node(lat, lon)
                coord_to_node[key] = node_id
            return coord_to_node[key]

        roads_processed = 0
        for _, road in drivable.iterrows():
            if road.geometry.geom_type != 'LineString':
                continue

            coords = list(road.geometry.coords)
            oneway = str(road.get('oneway', 'no')).lower() == 'yes'

            for i in range(len(coords) - 1):
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[i + 1]

                if not (-90 <= lat1 <= 90 and -180 <= lon1 <= 180):
                    continue
                if not (-90 <= lat2 <= 90 and -180 <= lon2 <= 180):
                    continue

                n1 = get_node(lat1, lon1)
                n2 = get_node(lat2, lon2)
                d = self.haversine(lat1, lon1, lat2, lon2)

                if d > 10000:
                    continue

                self.add_edge(n1, n2, d)
                if not oneway:
                    self.add_edge(n2, n1, d)

            roads_processed += 1

        print(
            f"Graph built: {len(self.nodes)} nodes, {sum(len(v) for v in self.adjacency.values())} edges from {roads_processed} roads.")

    def find_nearest_node(self, lat, lon):
        nearest, min_d = None, float('inf')
        for nid, (nlat, nlon) in self.nodes.items():
            d = self.haversine(lat, lon, nlat, nlon)
            if d < min_d:
                nearest, min_d = nid, d
        return nearest

    def is_connected(self, start, end):
        visited = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node == end:
                return True
            if node not in visited:
                visited.add(node)
                for neighbor, _ in self.adjacency.get(node, []):
                    if neighbor not in visited:
                        stack.append(neighbor)
        return False

class SearchAlgorithms:
    def __init__(self, graph):
        self.graph = graph

    def dijkstra(self, start, end):
        t0 = time.time()#ehleh tsagiig hemjine her udaan ajillahiig tootsno
        dist = {n: float("inf") for n in self.graph.nodes}#buh zaig hyazgaargui gej ehluuleh
        prev = {n: None for n in self.graph.nodes}#umnuh oroig hadgalna
        dist[start] = 0#ehleh oroin zai 0
        pq = [(0, start)]#hamgiin baga zaitai oroig turuulj shalgah buh oroig daraalald oruulah

        nodes_visited = 0
        while pq:
            d, u = heapq.heappop(pq)#queue ees hamgiin bogino zaitai oroig avna
            nodes_visited += 1

            if u == end:
                break
#u oroin hursh buriig shalgana v->hursh oroi w->zai nd->shine tootsoolson niit zai
            for v, w in self.graph.adjacency.get(u, []):
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        path, node = [], end
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()

        if not path or path[0] != start:
            return [], float('inf'), round(time.time() - t0, 3), nodes_visited

        return path, dist[end], round(time.time() - t0, 3), nodes_visited

    def bfs(self, start, end):
        t0 = time.time()
        if start not in self.graph.adjacency or end not in self.graph.adjacency:
            return [], float('inf'), round(time.time() - t0, 3), 0

        visited = set([start])
        queue = deque([(start, [start])])
        nodes_visited = 1

        while queue:
            node, path = queue.popleft()
            if node == end:
                distance = self.path_length(path)
                return path, distance, round(time.time() - t0, 3), nodes_visited #anhnii oldson zam ni hamgiin bogino
            for neigh, w in self.graph.adjacency.get(node, []):
                if neigh not in visited:
                    visited.add(neigh)
                    nodes_visited += 1
                    queue.append((neigh, path + [neigh]))

        return [], float('inf'), round(time.time() - t0, 3), nodes_visited

    def dfs(self, start, end, max_depth=5000):
        t0 = time.time()

        # Эхлээд холбогдсон эсэхийг шалгах
        if not self.graph.is_connected(start, end):
            return [], float('inf'), round(time.time() - t0, 3), 0

        best_path = None
        best_distance = float('inf')
        nodes_visited = 0

        def dfs_recursive(current, path, visited, depth):
            nonlocal best_path, best_distance, nodes_visited

            nodes_visited += 1

            if depth > max_depth:
                return

            if current == end:
                current_distance = self.path_length(path)
                if current_distance < best_distance:
                    best_path = path.copy()
                    best_distance = current_distance
                return

            visited.add(current)

            # Хөршүүдийг эрэмбэлэх (зайгаар)
            neighbors = self.graph.adjacency.get(current, [])
            # Богино зайтай хөршүүдээс эхлэх
            neighbors.sort(key=lambda x: x[1])

            for neighbor, weight in neighbors:
                if neighbor not in visited:
                    dfs_recursive(neighbor, path + [neighbor], visited.copy(), depth + 1)

        dfs_recursive(start, [start], set(), 0)

        if best_path:
            return best_path, best_distance, round(time.time() - t0, 3), nodes_visited

        return [], float('inf'), round(time.time() - t0, 3), nodes_visited

    def iterative_deepening_dfs(self, start, end, max_depth=200):
        t0 = time.time()

        # Алхам алхмаар гүний хязгаарыг нэмэгдүүлэх
        for depth in range(1, max_depth + 1, 10):  # 10 алхам тутамд нэмэгдүүлэх
            visited = set()
            stack = [(start, [start], 0)]  # (node, path, current_depth)
            best_path = None
            best_distance = float('inf')
            nodes_visited = 0

            while stack:
                node, path, current_depth = stack.pop()
                nodes_visited += 1

                if node == end:
                    current_distance = self.path_length(path)
                    if current_distance < best_distance:
                        best_path = path
                        best_distance = current_distance
                    continue

                if current_depth >= depth:
                    continue

                if node not in visited:
                    visited.add(node)
                    # Хөршүүдийг нэмэх (зайгаар эрэмбэлсэн)
                    neighbors = self.graph.adjacency.get(node, [])
                    neighbors.sort(key=lambda x: x[1])  # Богино зайгаар эрэмбэлэх

                    for neighbor, weight in neighbors:
                        if neighbor not in visited:
                            stack.append((neighbor, path + [neighbor], current_depth + 1))

            if best_path:
                print(f"IDDFS found path at depth {depth}")
                return best_path, best_distance, round(time.time() - t0, 3), nodes_visited

        return [], float('inf'), round(time.time() - t0, 3), nodes_visited

    def dfs_optimized(self, start, end, max_depth=10000, timeout=30):
        t0 = time.time()

        if not self.graph.is_connected(start, end):
            return [], float('inf'), round(time.time() - t0, 3), 0

        best_path = None
        best_distance = float('inf')
        nodes_visited = 0

        def dfs_recursive(current, path, visited, depth):
            nonlocal best_path, best_distance, nodes_visited

            # Timeout шалгах
            if time.time() - t0 > timeout:
                return True  # timeout occurred

            nodes_visited += 1

            # Гүний хязгаар
            if depth > max_depth:
                return False

            if current == end:
                current_distance = self.path_length(path)
                if current_distance < best_distance:
                    best_path = path.copy()
                    best_distance = current_distance
                return False

            visited.add(current)

            # Хөршүүдийг эрэмбэлэх
            neighbors = self.graph.adjacency.get(current, [])
            # Ойрхон хөршүүдээс эхлэх
            neighbors.sort(key=lambda x: x[1])

            for neighbor, weight in neighbors:
                if neighbor not in visited:
                    timeout_occurred = dfs_recursive(neighbor, path + [neighbor], visited.copy(), depth + 1)
                    if timeout_occurred:
                        return True

            return False

        timeout_occurred = dfs_recursive(start, [start], set(), 0)

        if timeout_occurred:
            print("!!! DFS timeout reached")

        if best_path:
            return best_path, best_distance, round(time.time() - t0, 3), nodes_visited

        return [], float('inf'), round(time.time() - t0, 3), nodes_visited

    def dfs_non_recursive(self, start, end, max_depth=10000):
        t0 = time.time()

        best_path = None
        best_distance = float('inf')
        nodes_visited = 0

        stack = [(start, [start], 0)]  # (node, path, depth)
        visited_global = set()

        while stack:
            node, path, depth = stack.pop()
            nodes_visited += 1

            if node == end:
                current_distance = self.path_length(path)
                if current_distance < best_distance:
                    best_path = path
                    best_distance = current_distance
                continue

            if depth > max_depth:
                continue

            if node not in visited_global:
                visited_global.add(node)

                # Хөршүүдийг нэмэх
                neighbors = self.graph.adjacency.get(node, [])
                # Алслагдсан хөршүүдийг сүүлийн ээлжинд нэмэх
                neighbors.sort(key=lambda x: x[1], reverse=True)

                for neighbor, weight in neighbors:
                    if neighbor not in visited_global:
                        stack.append((neighbor, path + [neighbor], depth + 1))

        if best_path:
            return best_path, best_distance, round(time.time() - t0, 3), nodes_visited

        return [], float('inf'), round(time.time() - t0, 3), nodes_visited

    def path_length(self, path):
        total = 0
        for i in range(len(path) - 1):
            for neigh, w in self.graph.adjacency.get(path[i], []):
                if neigh == path[i + 1]:
                    total += w
                    break
        return total

graph = GraphBuilder()
try:
    graph.load_osm_data("gis_osm_roads_free_1.shp")
    print("OSM data loaded successfully")
except Exception as e:
    print(f"Error loading OSM data: {e}")
    print("Creating comprehensive test data...")

    # Дэлгэрэнгүй туршилтын өгөгдөл үүсгэх
    graph = GraphBuilder()

    # Улаанбаатарын гол байршилууд
    test_locations = [
        (47.9185, 106.9172, "Сүхбаатар талбай"),
        (47.9204, 106.9178, "Улаанбаатар банк"),
        (47.9178, 106.9056, "ХУД"),
        (47.9089, 106.9125, "Найрамдал цэнгэлдэх"),
        (47.9123, 106.9241, "Хан-Уул"),
        (47.8996, 106.9187, "Баянгол"),
        (47.9256, 106.9321, "Сонгинохайрхан"),
        (47.8932, 106.8923, "Толгойт"),
        (47.9315, 106.9087, "Бага тойруу"),
        (47.9021, 106.8764, "Их тойруу")
    ]

    # Цэгүүдийг нэмэх
    node_ids = {}
    for i, (lat, lon, name) in enumerate(test_locations):
        node_id = graph.add_node(lat, lon)
        node_ids[name] = node_id
        print(f" {name}: node {node_id}")

    # Замын холболтууд
    test_connections = [
        ("Сүхбаатар талбай", "Улаанбаатар банк", 150),
        ("Сүхбаатар талбай", "ХУД", 800),
        ("Сүхбаатар талбай", "Хан-Уул", 1200),
        ("Улаанбаатар банк", "ХУД", 700),
        ("Улаанбаатар банк", "Бага тойруу", 2500),
        ("ХУД", "Найрамдал цэнгэлдэх", 600),
        ("ХУД", "Баянгол", 1800),
        ("Найрамдал цэнгэлдэх", "Хан-Уул", 900),
        ("Найрамдал цэнгэлдэх", "Баянгол", 1100),
        ("Хан-Уул", "Сонгинохайрхан", 1400),
        ("Хан-Уул", "Бага тойруу", 2100),
        ("Баянгол", "Толгойт", 800),
        ("Баянгол", "Их тойруу", 1500),
        ("Сонгинохайрхан", "Бага тойруу", 1200),
        ("Толгойт", "Их тойруу", 700),
        ("Бага тойруу", "Их тойруу", 3500)
    ]

    for loc1, loc2, dist in test_connections:
        if loc1 in node_ids and loc2 in node_ids:
            graph.add_edge(node_ids[loc1], node_ids[loc2], dist)
            graph.add_edge(node_ids[loc2], node_ids[loc1], dist)
            print(f" {loc1} ↔ {loc2}: {dist}m")

search = SearchAlgorithms(graph)


# === 4. FLASK ROUTES ===
@app.route("/")
def home():
    return render_template("map.html")


@app.route("/api/path", methods=["POST"])
def get_path():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data received"})

        start_node = graph.find_nearest_node(data["start_lat"], data["start_lon"])
        end_node = graph.find_nearest_node(data["end_lat"], data["end_lon"])
        algo = data["algorithm"]

        print(f"Path request: {algo} from {start_node} to {end_node}")

        # Safety check for isolated nodes
        if not graph.adjacency.get(start_node):
            return jsonify({"success": False, "error": f"Start node {start_node} has no neighbors"})
        if not graph.adjacency.get(end_node):
            return jsonify({"success": False, "error": f"End node {end_node} has no neighbors"})

        # Холбогдсон эсэхийг шалгах
        if not graph.is_connected(start_node, end_node):
            return jsonify({"success": False, "error": "Start and end nodes are not connected"})

        if algo == "dijkstra":
            path, dist, t, nodes_visited = search.dijkstra(start_node, end_node)
        elif algo == "bfs":
            path, dist, t, nodes_visited = search.bfs(start_node, end_node)
        elif algo == "dfs":
            # DFS-ийн 2 хувилбарыг турших
            path, dist, t, nodes_visited = search.iterative_deepening_dfs(start_node, end_node)
            if not path:
                path, dist, t, nodes_visited = search.dfs(start_node, end_node)
        else:
            return jsonify({"success": False, "error": "Invalid algorithm"})

        if not path or dist == float("inf"):
            return jsonify({"success": False, "error": "No path found between selected points"})

        coords = [[graph.nodes[n][0], graph.nodes[n][1]] for n in path]

        print(f"Path found: {len(path)} nodes, {dist:.1f}m, {t}s, visited {nodes_visited} nodes")

        return jsonify({
            "success": True,
            "path": coords,
            "distance": round(dist, 1),
            "time": t,
            "nodes_visited": nodes_visited,
            "path_nodes": len(path)
        })

    except Exception as e:
        print(f"!!! Error in path calculation: {e}")
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500


@app.route("/api/graph_info", methods=["GET"])
def graph_info():
    """Return graph information"""
    total_edges = sum(len(v) for v in graph.adjacency.values())
    return jsonify({
        "nodes": len(graph.nodes),
        "edges": total_edges,
        "density": f"{(total_edges / len(graph.nodes)):.2f}" if graph.nodes else "0",
        "bounds": {
            "min_lat": min(lat for lat, lon in graph.nodes.values()) if graph.nodes else 47.8,
            "max_lat": max(lat for lat, lon in graph.nodes.values()) if graph.nodes else 48.0,
            "min_lon": min(lon for lat, lon in graph.nodes.values()) if graph.nodes else 106.8,
            "max_lon": max(lon for lat, lon in graph.nodes.values()) if graph.nodes else 107.0
        }
    })


@app.route("/api/debug_connection", methods=["POST"])
def debug_connection():
    """Хоёр цэгийн холболтыг шалгах"""
    data = request.get_json()
    start_node = graph.find_nearest_node(data["start_lat"], data["start_lon"])
    end_node = graph.find_nearest_node(data["end_lat"], data["end_lon"])

    return jsonify({
        "start_node": start_node,
        "end_node": end_node,
        "start_has_neighbors": bool(graph.adjacency.get(start_node)),
        "end_has_neighbors": bool(graph.adjacency.get(end_node)),
        "connected": graph.is_connected(start_node, end_node),
        "start_neighbors": len(graph.adjacency.get(start_node, [])),
        "end_neighbors": len(graph.adjacency.get(end_node, []))
    })


if __name__ == "__main__":
    print("Starting Flask server...")
    print("Graph Statistics:")
    print(f"   - Nodes: {len(graph.nodes)}")
    print(f"   - Edges: {sum(len(v) for v in graph.adjacency.values())}")
    print(f"   - Node 1 neighbors: {len(graph.adjacency.get(1, []))}")
    print(f"   - Node 2 neighbors: {len(graph.adjacency.get(2, []))}")
    print("Server running on http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')