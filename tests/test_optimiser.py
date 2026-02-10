import pytest
import random
from models import Control, Path
from graph import build_graph
from optimiser import (
    edge_cost,
    edge_distance_2d,
    build_path_from_ids,
    greedy_construct,
    two_opt_improve,
    try_insert_unvisited,
    optimise,
)


def _make_control(label, easting, northing, elevation, points=None):
    c = Control(0, 0, label=label, elevation=elevation, points=points)
    c.easting = easting
    c.northing = northing
    return c


@pytest.fixture
def controls():
    home = _make_control("0", 0, 0, 100, points=0)
    c1 = _make_control("50", 100, 0, 110, points=50)
    c2 = _make_control("30", 200, 0, 105, points=30)
    c3 = _make_control("80", 0, 100, 120, points=80)
    c4 = _make_control("40", 100, 100, 115, points=40)
    return [home, c1, c2, c3, c4]


@pytest.fixture
def graph(controls):
    return build_graph(controls, lambda a, b: edge_cost(a, b, 10.0))


@pytest.fixture
def seeded():
    random.seed(42)


class TestEdgeCost:
    def test_flat_terrain(self):
        c1 = _make_control("0", 0, 0, 100)
        c2 = _make_control("50", 3, 4, 100)
        cost = edge_cost(c1, c2, 10.0)
        assert cost == pytest.approx(5.0)

    def test_uphill_adds_climb(self):
        c1 = _make_control("0", 0, 0, 100)
        c2 = _make_control("50", 0, 100, 150)
        cost = edge_cost(c1, c2, 10.0)
        assert cost == pytest.approx(100.0 + 10.0 * 50.0)

    def test_downhill_no_climb_penalty(self):
        c1 = _make_control("0", 0, 0, 150)
        c2 = _make_control("50", 0, 100, 100)
        cost = edge_cost(c1, c2, 10.0)
        assert cost == pytest.approx(100.0)

    def test_zero_climb_factor(self):
        c1 = _make_control("0", 0, 0, 100)
        c2 = _make_control("50", 0, 100, 200)
        cost = edge_cost(c1, c2, 0.0)
        assert cost == pytest.approx(100.0)


class TestEdgeDistance2d:
    def test_basic(self):
        c1 = _make_control("0", 0, 0, 100)
        c2 = _make_control("50", 3, 4, 200)
        assert edge_distance_2d(c1, c2) == pytest.approx(5.0)


class TestBuildPathFromIds:
    def test_correct_totals(self, controls, graph):
        home = controls[0]
        c1 = controls[1]
        route = [home.id, c1.id, home.id]
        path = build_path_from_ids(route, graph, 10.0)
        assert len(path.controls) == 3
        assert path.controls[0] is home
        assert path.controls[1] is c1
        assert path.controls[2] is home
        assert path.points == 50
        assert path.distance_2d > 0

    def test_single_node(self, controls, graph):
        home = controls[0]
        path = build_path_from_ids([home.id], graph, 10.0)
        assert path.points == 0
        assert path.distance_2d == 0.0


class TestGreedyConstruct:
    def test_starts_and_ends_at_home(self, controls, graph, seeded):
        home = controls[0]
        route, _ = greedy_construct(graph, home.id, 10000, 0, 3, 10.0)
        assert route[0] == home.id
        assert route[-1] == home.id

    def test_respects_budget(self, controls, graph, seeded):
        home = controls[0]
        max_cost = 10000
        route, cost = greedy_construct(graph, home.id, max_cost, 0, 3, 10.0)
        path = build_path_from_ids(route, graph, 10.0)
        assert path.total_cost <= max_cost + 1e-6

    def test_tight_budget_minimal_route(self, controls, graph, seeded):
        home = controls[0]
        route, _ = greedy_construct(graph, home.id, 1.0, 0, 3, 10.0)
        assert route[0] == home.id
        assert route[-1] == home.id
        path = build_path_from_ids(route, graph, 10.0)
        assert path.points == 0

    def test_collects_points(self, controls, graph, seeded):
        home = controls[0]
        route, _ = greedy_construct(graph, home.id, 10000, 0, 5, 10.0)
        path = build_path_from_ids(route, graph, 10.0)
        assert path.points > 0

    def test_no_duplicate_visits(self, controls, graph, seeded):
        home = controls[0]
        route, _ = greedy_construct(graph, home.id, 10000, 0, 5, 10.0)
        interior = route[1:-1]
        assert len(interior) == len(set(interior))


class TestTwoOptImprove:
    def test_does_not_increase_cost(self, controls, graph, seeded):
        home = controls[0]
        route, _ = greedy_construct(graph, home.id, 10000, 0, 5, 10.0)
        original_path = build_path_from_ids(route, graph, 10.0)
        improved_route, improved_cost = two_opt_improve(route, graph, 10.0, 50)
        improved_path = build_path_from_ids(improved_route, graph, 10.0)
        assert improved_path.total_cost <= original_path.total_cost + 1e-6

    def test_preserves_home(self, controls, graph, seeded):
        home = controls[0]
        route, _ = greedy_construct(graph, home.id, 10000, 0, 5, 10.0)
        improved, _ = two_opt_improve(route, graph, 10.0)
        assert improved[0] == home.id
        assert improved[-1] == home.id

    def test_preserves_visited_set(self, controls, graph, seeded):
        home = controls[0]
        route, _ = greedy_construct(graph, home.id, 10000, 0, 5, 10.0)
        improved, _ = two_opt_improve(route, graph, 10.0)
        assert set(improved) == set(route)

    def test_short_route_unchanged(self, controls, graph):
        home = controls[0]
        c1 = controls[1]
        route = [home.id, c1.id, home.id]
        improved, _ = two_opt_improve(route, graph, 10.0)
        assert improved == route


class TestTryInsertUnvisited:
    def test_inserts_when_budget_allows(self, controls, graph, seeded):
        home = controls[0]
        c1 = controls[1]
        route = [home.id, c1.id, home.id]
        expanded = try_insert_unvisited(route, graph, 100000, 10.0)
        assert len(expanded) >= len(route)

    def test_does_not_exceed_budget(self, controls, graph, seeded):
        home = controls[0]
        c1 = controls[1]
        route = [home.id, c1.id, home.id]
        initial_path = build_path_from_ids(route, graph, 10.0)
        max_cost = initial_path.total_cost + 500.0
        expanded = try_insert_unvisited(route, graph, max_cost, 10.0)
        path = build_path_from_ids(expanded, graph, 10.0)
        assert path.total_cost <= max_cost + 1e-6


class TestOptimise:
    def test_returns_valid_path(self, controls, graph, seeded):
        home = controls[0]
        path = optimise(graph, home.id, 10000, 0, 20, 3, 10.0, 10)
        assert path is not None
        assert path.points > 0

    def test_route_starts_ends_home(self, controls, graph, seeded):
        home = controls[0]
        path = optimise(graph, home.id, 10000, 0, 20, 3, 10.0, 10)
        assert path.controls[0].label == "0"
        assert path.controls[-1].label == "0"

    def test_tight_budget(self, controls, graph, seeded):
        home = controls[0]
        path = optimise(graph, home.id, 250, 0, 20, 3, 10.0, 10)
        if path is not None:
            assert path.distance_2d <= 250 + 1e-6
