"""맵 노드 그래프: 최단 경로와 이동 시간."""
import heapq
import math


class MapGraph:
    def __init__(self, map_json):
        self.nodes = map_json["nodes"]
        self.adj = {n: {} for n in self.nodes}
        for a, b, factor in map_json["edges"]:
            pa, pb = self.nodes[a], self.nodes[b]
            d = math.dist((pa["x"], pa["y"]), (pb["x"], pb["y"])) * factor
            self.adj[a][b] = d
            self.adj[b][a] = d
        self._dist, self._prev = {}, {}
        for n in self.nodes:
            self._dist[n], self._prev[n] = self._dijkstra(n)

    def _dijkstra(self, src):
        dist, prev = {src: 0.0}, {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in self.adj[u].items():
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    def dist(self, a, b):
        return self._dist[a][b]

    def path(self, a, b):
        """a 다음 노드부터 b까지(포함). a == b면 빈 목록."""
        out, cur = [], b
        while cur != a:
            out.append(cur)
            cur = self._prev[a][cur]
        return out[::-1]

    def travel_s(self, a, b, ms):
        return self.dist(a, b) / ms

    def hops(self, a, b):
        return len(self.path(a, b))

    def is_connected(self):
        start = next(iter(self.nodes))
        return len(self._dist[start]) == len(self.nodes)
