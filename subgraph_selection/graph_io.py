from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Hashable, Iterable, Mapping, MutableMapping, Tuple

import networkx as nx


NodeId = Hashable
EdgeKey = Tuple[NodeId, NodeId]


DEFAULT_FLOOD_TRANSFORM = {
    0.0: 0.0,
    0.2: 1.0,
    0.6: 4.0,
    1.0: 9.0,
}

DEFAULT_LANDSLIDE_TRANSFORM = {
    0.0: 0.0,
    0.3: 1.0,
    0.6: 4.0,
    1.0: 9.0,
}

DEFAULT_RI_MULTIPLIERS = {
    "RI1": 1.00,
    "RI2": 1.25,
    "RI3": 1.60,
    "RI4": 2.10,
    "RI5": 2.80,
}


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def canonical_edge(u: NodeId, v: NodeId) -> EdgeKey:
    return (u, v) if str(u) <= str(v) else (v, u)


def _to_undirected_simple(raw_graph: nx.Graph) -> nx.Graph:
    graph_u = nx.Graph()
    graph_u.graph.update(dict(raw_graph.graph))

    for node, data in raw_graph.nodes(data=True):
        graph_u.add_node(node, **dict(data))

    if raw_graph.is_multigraph():
        iterator = raw_graph.edges(keys=True, data=True)
        for u, v, _, data in iterator:
            payload = dict(data)
            candidate_time = safe_float(payload.get("travel_time_min", math.inf), math.inf)
            if graph_u.has_edge(u, v):
                previous_time = safe_float(graph_u[u][v].get("travel_time_min", math.inf), math.inf)
                if candidate_time >= previous_time:
                    continue
            graph_u.add_edge(u, v, **payload)
    else:
        for u, v, data in raw_graph.edges(data=True):
            payload = dict(data)
            candidate_time = safe_float(payload.get("travel_time_min", math.inf), math.inf)
            if graph_u.has_edge(u, v):
                previous_time = safe_float(graph_u[u][v].get("travel_time_min", math.inf), math.inf)
                if candidate_time >= previous_time:
                    continue
            graph_u.add_edge(u, v, **payload)

    return graph_u


def load_simple_graph(graphml_path: str | Path) -> nx.Graph:
    graph_raw = nx.read_graphml(Path(graphml_path))
    graph_u = _to_undirected_simple(graph_raw)
    largest_cc = max(nx.connected_components(graph_u), key=len)
    return graph_u.subgraph(largest_cc).copy()


@dataclass(frozen=True)
class RiskModel:
    flood_attr: str = "flood_hazard"
    landslide_attr: str = "landslide_hazard"
    time_attr: str = "travel_time_min"
    length_attr: str = "length"
    flood_weight: float = 1.0
    landslide_weight: float = 1.0
    flood_transform: Mapping[float, float] = field(default_factory=lambda: DEFAULT_FLOOD_TRANSFORM.copy())
    landslide_transform: Mapping[float, float] = field(default_factory=lambda: DEFAULT_LANDSLIDE_TRANSFORM.copy())
    ri_multipliers: Mapping[str, float] = field(default_factory=lambda: DEFAULT_RI_MULTIPLIERS.copy())

    def _lookup_transform(self, value: float, table: Mapping[float, float]) -> float:
        if not table:
            return float(value)
        if value in table:
            return float(table[value])
        nearest_key = min(table.keys(), key=lambda key: abs(float(key) - float(value)))
        return float(table[nearest_key])

    def edge_time_cost(self, edge_data: MutableMapping[str, object]) -> float:
        cost = safe_float(edge_data.get(self.time_attr, math.nan), math.nan)
        if math.isfinite(cost) and cost > 0:
            return cost

        length_m = max(safe_float(edge_data.get(self.length_attr, 1.0), 1.0), 1e-3)
        return (length_m / 8.33) / 60.0

    def edge_hazard_components(self, edge_data: MutableMapping[str, object]) -> Tuple[float, float]:
        flood_score = safe_float(edge_data.get(self.flood_attr, 0.0), 0.0)
        landslide_score = safe_float(edge_data.get(self.landslide_attr, 0.0), 0.0)
        flood_term = self._lookup_transform(flood_score, self.flood_transform)
        landslide_term = self._lookup_transform(landslide_score, self.landslide_transform)
        return float(flood_term), float(landslide_term)

    def edge_hazard_cost(self, edge_data: MutableMapping[str, object], rain_key: str) -> float:
        length_m = max(safe_float(edge_data.get(self.length_attr, 1.0), 1.0), 1e-3)
        flood_term, landslide_term = self.edge_hazard_components(edge_data)
        severity = float(self.ri_multipliers.get(rain_key, 1.0))
        return length_m * severity * (
            self.flood_weight * flood_term + self.landslide_weight * landslide_term
        )


def graphml_safe_graph(graph: nx.Graph) -> nx.Graph:
    converted = graph.__class__()
    for key, value in graph.graph.items():
        if key in {"node_default", "edge_default"}:
            converted.graph[key] = value if isinstance(value, dict) else {}
        else:
            converted.graph[key] = _to_graphml_value(value)

    for node, data in graph.nodes(data=True):
        converted.add_node(node, **{key: _to_graphml_value(value) for key, value in data.items()})

    if graph.is_multigraph():
        for u, v, k, data in graph.edges(keys=True, data=True):
            converted.add_edge(u, v, key=k, **{key: _to_graphml_value(value) for key, value in data.items()})
    else:
        for u, v, data in graph.edges(data=True):
            converted.add_edge(u, v, **{key: _to_graphml_value(value) for key, value in data.items()})

    return converted


def _to_graphml_value(value):
    if value is None:
        return ""
    if isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return ""
        return value
    if isinstance(value, (list, tuple, set, dict)):
        return str(value)
    if hasattr(value, "wkt"):
        return value.wkt
    return str(value)

