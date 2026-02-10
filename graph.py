import copy
import numpy as np


class Graph:
    def __init__(self):
        self.edges = {}  # {from_id: {to_id: cost}}
        self.controls_by_id = {}  # {id: Control}

    def add_node(self, control):
        self.controls_by_id[control.id] = control
        if control.id not in self.edges:
            self.edges[control.id] = {}

    def add_edge(self, from_ctrl, to_ctrl, cost):
        if from_ctrl.id not in self.edges:
            self.edges[from_ctrl.id] = {}
        self.edges[from_ctrl.id][to_ctrl.id] = cost

    def get_cost(self, from_id, to_id):
        return self.edges[from_id][to_id]

    def get_control(self, ctrl_id):
        return self.controls_by_id[ctrl_id]

    def get_neighbors(self, ctrl_id):
        """Returns list of (neighbor_id, cost) sorted by cost ascending."""
        neighbors = self.edges.get(ctrl_id, {})
        return sorted(neighbors.items(), key=lambda x: x[1])

    def copy(self):
        new_graph = Graph()
        new_graph.edges = copy.deepcopy(self.edges)
        new_graph.controls_by_id = self.controls_by_id  # shared ref is fine, controls don't mutate
        return new_graph

    def __str__(self):
        lines = ["Graph:"]
        for node_id, neighbors in self.edges.items():
            ctrl = self.controls_by_id.get(node_id)
            label = ctrl.label if ctrl else str(node_id)
            neighbor_strs = []
            for nid, cost in neighbors.items():
                nctrl = self.controls_by_id.get(nid)
                nlabel = nctrl.label if nctrl else str(nid)
                neighbor_strs.append(f"{nlabel}: {cost:.1f}")
            lines.append(f"  {label} -> {{{', '.join(neighbor_strs)}}}")
        return "\n".join(lines)


def build_graph(controls, cost_fn):
    """Build a complete graph from a list of controls using cost_fn(ctrl_a, ctrl_b) -> float."""
    graph = Graph()
    for ctrl in controls:
        graph.add_node(ctrl)
    for orig in controls:
        for dest in controls:
            if orig.id != dest.id:
                cost = cost_fn(orig, dest)
                graph.add_edge(orig, dest, cost)
    return graph
