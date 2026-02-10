import random
from math import sqrt
from models import Path


def edge_cost(ctrl_a, ctrl_b, climb_factor):
    """Compute weighted cost between two controls including elevation."""
    dist_2d = ctrl_a.geo_distance_2d(ctrl_b)
    climb = ctrl_a.get_climb(ctrl_b)
    return dist_2d + climb_factor * climb


def edge_distance_2d(ctrl_a, ctrl_b):
    return ctrl_a.geo_distance_2d(ctrl_b)


def build_path_from_ids(control_ids, graph, climb_factor):
    """Reconstruct a Path object from an ordered list of control IDs."""
    path = Path()
    for i, cid in enumerate(control_ids):
        ctrl = graph.get_control(cid)
        path.controls.append(ctrl)
        if i == 0:
            continue
        prev = graph.get_control(control_ids[i - 1])
        path.distance_2d += edge_distance_2d(prev, ctrl)
        path.total_cost += edge_cost(prev, ctrl, climb_factor)
        path.total_climb += prev.get_climb(ctrl)
        path.points += ctrl.points if ctrl.points else 0
    return path


def greedy_construct(graph, home_id, max_cost, min_points_before_return, top_k, climb_factor):
    """Budget-aware greedy construction with feasibility-to-return check.

    At each step, picks among the top-K candidates by points/cost efficiency,
    ensuring we can still return home within budget.
    """
    visited = {home_id}
    route = [home_id]
    current_cost = 0.0
    current_points = 0

    while True:
        current_id = route[-1]
        current_ctrl = graph.get_control(current_id)
        candidates = []

        for neighbor_id, _ in graph.get_neighbors(current_id):
            if neighbor_id in visited:
                continue
            neighbor_ctrl = graph.get_control(neighbor_id)
            cost_to_neighbor = edge_cost(current_ctrl, neighbor_ctrl, climb_factor)
            cost_neighbor_to_home = edge_cost(neighbor_ctrl, graph.get_control(home_id), climb_factor)

            if current_cost + cost_to_neighbor + cost_neighbor_to_home > max_cost:
                continue

            pts = neighbor_ctrl.points if neighbor_ctrl.points else 0
            if cost_to_neighbor > 0:
                efficiency = pts / cost_to_neighbor
            else:
                efficiency = float("inf") if pts > 0 else 0

            candidates.append((neighbor_id, efficiency, cost_to_neighbor))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:top_k]

        if current_points < min_points_before_return:
            top = [c for c in top if graph.get_control(c[0]).label != "0"]
            if not top:
                top = candidates[:top_k]

        weights = [max(c[1], 0.01) for c in top]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        chosen = random.choices(top, weights=weights, k=1)[0]
        chosen_id, _, chosen_cost = chosen

        route.append(chosen_id)
        visited.add(chosen_id)
        current_cost += chosen_cost
        ctrl = graph.get_control(chosen_id)
        current_points += ctrl.points if ctrl.points else 0

    if route[-1] != home_id:
        current_ctrl = graph.get_control(route[-1])
        home_ctrl = graph.get_control(home_id)
        cost_home = edge_cost(current_ctrl, home_ctrl, climb_factor)
        current_cost += cost_home
        route.append(home_id)

    return route, current_cost


def two_opt_improve(route, graph, climb_factor, max_iterations=100):
    """Apply 2-opt swaps to reduce total route cost while keeping the same visited set.

    The route starts and ends at home (route[0] == route[-1]).
    We only swap the interior segment (indices 1 to len-2).
    """

    def route_cost(r):
        total = 0.0
        for i in range(len(r) - 1):
            total += edge_cost(graph.get_control(r[i]), graph.get_control(r[i + 1]), climb_factor)
        return total

    best_route = route[:]
    best_cost = route_cost(best_route)
    n = len(best_route)

    if n < 4:
        return best_route, best_cost

    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_route = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                new_cost = route_cost(new_route)
                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break

    return best_route, best_cost


def try_insert_unvisited(route, graph, max_cost, climb_factor):
    """Try inserting high-value unvisited controls into the route if they fit budget."""
    visited = set(route)
    all_ids = set(graph.controls_by_id.keys())
    unvisited = all_ids - visited

    def route_cost(r):
        total = 0.0
        for i in range(len(r) - 1):
            total += edge_cost(graph.get_control(r[i]), graph.get_control(r[i + 1]), climb_factor)
        return total

    current_cost = route_cost(route)

    unvisited_by_points = sorted(
        unvisited,
        key=lambda cid: graph.get_control(cid).points or 0,
        reverse=True,
    )

    best_route = route[:]
    for uid in unvisited_by_points:
        u_ctrl = graph.get_control(uid)
        if not u_ctrl.points:
            continue
        best_insert_cost = float("inf")
        best_pos = None
        for pos in range(1, len(best_route)):
            prev_ctrl = graph.get_control(best_route[pos - 1])
            next_ctrl = graph.get_control(best_route[pos])
            old_edge = edge_cost(prev_ctrl, next_ctrl, climb_factor)
            new_edge = edge_cost(prev_ctrl, u_ctrl, climb_factor) + edge_cost(u_ctrl, next_ctrl, climb_factor)
            delta = new_edge - old_edge
            if delta < best_insert_cost:
                best_insert_cost = delta
                best_pos = pos
        if best_pos is not None and current_cost + best_insert_cost <= max_cost:
            best_route.insert(best_pos, uid)
            current_cost += best_insert_cost

    return best_route


def optimise(graph, home_id, max_distance_m, min_points_before_return, iterations,
             top_k, climb_factor, two_opt_iters=100, mode="max_efficiency",
             on_path=None):
    """Run multi-start budget-aware greedy + 2-opt optimiser.

    Args:
        mode: "max_points" to maximise total points, "max_efficiency" to maximise pts/m.
        on_path: optional callback(path, iteration, is_new_best) called after each iteration
                 for live visualisation. Return False from callback to stop early.

    Returns the best Path found.
    """
    best_path = None
    best_score = 0.0

    for i in range(iterations):
        route, cost = greedy_construct(
            graph, home_id, max_distance_m, min_points_before_return, top_k, climb_factor
        )

        route = try_insert_unvisited(route, graph, max_distance_m, climb_factor)

        route, cost = two_opt_improve(route, graph, climb_factor, max_iterations=two_opt_iters)

        path = build_path_from_ids(route, graph, climb_factor)

        if mode == "max_points":
            score = path.points
        else:
            score = path.points / path.distance_2d if path.distance_2d > 0 else 0

        is_new_best = score > best_score
        if is_new_best:
            best_score = score
            best_path = path
            eff = path.points / path.distance_2d if path.distance_2d > 0 else 0
            print(f"  [{i+1}/{iterations}] New best: {path.points} pts, "
                  f"{path.distance_2d/1000:.2f} km, {path.total_climb:.0f}m climb, "
                  f"eff={eff:.4f}")

        if on_path is not None:
            if on_path(path, i, is_new_best) is False:
                break

    return best_path
