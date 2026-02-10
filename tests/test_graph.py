import pytest
from models import Control
from graph import Graph, build_graph


def make_control(label, easting=0, northing=0, elevation=100):
    c = Control(0, 0, label=label, elevation=elevation)
    c.easting, c.northing = easting, northing
    return c


class TestGraph:
    def test_no_shared_state(self):
        g1 = Graph()
        g2 = Graph()
        assert g1.edges is not g2.edges
        assert g1.controls_by_id is not g2.controls_by_id

    def test_add_node(self):
        g = Graph()
        c = make_control("50")
        g.add_node(c)
        assert c.id in g.edges
        assert g.edges[c.id] == {}
        assert g.controls_by_id[c.id] is c

    def test_add_edge(self):
        g = Graph()
        c1 = make_control("50", easting=0, northing=0)
        c2 = make_control("30", easting=100, northing=0)
        g.add_node(c1)
        g.add_node(c2)
        g.add_edge(c1, c2, 42.5)
        assert g.edges[c1.id][c2.id] == 42.5

    def test_get_cost(self):
        g = Graph()
        c1 = make_control("50")
        c2 = make_control("30")
        g.add_node(c1)
        g.add_node(c2)
        g.add_edge(c1, c2, 99.0)
        assert g.get_cost(c1.id, c2.id) == 99.0

    def test_get_control(self):
        g = Graph()
        c = make_control("50")
        g.add_node(c)
        assert g.get_control(c.id) is c

    def test_get_neighbors_sorted(self):
        g = Graph()
        c1 = make_control("0")
        c2 = make_control("50")
        c3 = make_control("30")
        for c in [c1, c2, c3]:
            g.add_node(c)
        g.add_edge(c1, c2, 200.0)
        g.add_edge(c1, c3, 50.0)
        neighbors = g.get_neighbors(c1.id)
        assert neighbors[0] == (c3.id, 50.0)
        assert neighbors[1] == (c2.id, 200.0)

    def test_get_neighbors_empty(self):
        g = Graph()
        c = make_control("0")
        g.add_node(c)
        assert g.get_neighbors(c.id) == []

    def test_copy_deep_copies_edges(self):
        g = Graph()
        c1 = make_control("0")
        c2 = make_control("50")
        g.add_node(c1)
        g.add_node(c2)
        g.add_edge(c1, c2, 10.0)

        g2 = g.copy()
        g2.edges[c1.id][c2.id] = 999.0
        assert g.edges[c1.id][c2.id] == 10.0

    def test_copy_shares_controls(self):
        g = Graph()
        c = make_control("50")
        g.add_node(c)
        g2 = g.copy()
        assert g2.controls_by_id[c.id] is c

    def test_str(self):
        g = Graph()
        c1 = make_control("0")
        c2 = make_control("50")
        g.add_node(c1)
        g.add_node(c2)
        g.add_edge(c1, c2, 10.0)
        s = str(g)
        assert "Graph:" in s
        assert "50" in s


class TestBuildGraph:
    def test_complete_graph(self):
        controls = [make_control(str(i * 10), easting=i * 100, northing=0) for i in range(4)]
        g = build_graph(controls, lambda a, b: a.geo_distance_2d(b))
        n = len(controls)
        total_edges = sum(len(v) for v in g.edges.values())
        assert total_edges == n * (n - 1)

    def test_uses_cost_fn(self):
        c1 = make_control("0", easting=0, northing=0)
        c2 = make_control("50", easting=100, northing=0)
        g = build_graph([c1, c2], lambda a, b: 42.0)
        assert g.get_cost(c1.id, c2.id) == 42.0
        assert g.get_cost(c2.id, c1.id) == 42.0

    def test_all_nodes_registered(self):
        controls = [make_control(str(i * 10)) for i in range(5)]
        g = build_graph(controls, lambda a, b: 1.0)
        for c in controls:
            assert c.id in g.controls_by_id
