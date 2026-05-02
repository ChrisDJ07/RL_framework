from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Dict, Hashable, Iterable, List, Mapping, Sequence, Set, Tuple

import networkx as nx

from utils.subgraph_selection.graph_io import EdgeKey, RiskModel, canonical_edge


NodeId = Hashable


@dataclass
class PairTradeoffDetail:
    rain_key: str
    time_path_time: float
    hazard_path_time: float
    time_path_hazard: float
    hazard_path_hazard: float
    delta_time_rel: float
    delta_hazard_rel: float
    divergence: float
    score: float
    union_edges: Set[EdgeKey] = field(default_factory=set)
    time_edges: Set[EdgeKey] = field(default_factory=set)
    hazard_edges: Set[EdgeKey] = field(default_factory=set)

    def to_json(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["union_edges"] = [list(edge) for edge in sorted(self.union_edges, key=lambda edge: (str(edge[0]), str(edge[1])))]
        payload["time_edges"] = [list(edge) for edge in sorted(self.time_edges, key=lambda edge: (str(edge[0]), str(edge[1])))]
        payload["hazard_edges"] = [list(edge) for edge in sorted(self.hazard_edges, key=lambda edge: (str(edge[0]), str(edge[1])))]
        return payload


@dataclass
class PairTradeoffAggregate:
    source: NodeId
    target: NodeId
    mean_score: float
    max_score: float
    mean_delta_time_rel: float
    mean_delta_hazard_rel: float
    mean_divergence: float
    union_edges: Set[EdgeKey]
    details: List[PairTradeoffDetail]

    def to_json(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "mean_score": self.mean_score,
            "max_score": self.max_score,
            "mean_delta_time_rel": self.mean_delta_time_rel,
            "mean_delta_hazard_rel": self.mean_delta_hazard_rel,
            "mean_divergence": self.mean_divergence,
            "union_edges": [list(edge) for edge in sorted(self.union_edges, key=lambda edge: (str(edge[0]), str(edge[1])))],
            "details": [detail.to_json() for detail in self.details],
        }


def path_edges(path: Sequence[NodeId]) -> Set[EdgeKey]:
    return {canonical_edge(path[i], path[i + 1]) for i in range(len(path) - 1)}


def path_cost(graph: nx.Graph, path: Sequence[NodeId], edge_cost_fn) -> float:
    total = 0.0
    for index in range(len(path) - 1):
        total += float(edge_cost_fn(graph[path[index]][path[index + 1]]))
    return total


def edge_divergence(edges_a: Set[EdgeKey], edges_b: Set[EdgeKey]) -> float:
    union = edges_a | edges_b
    if not union:
        return 0.0
    overlap = len(edges_a & edges_b) / len(union)
    return 1.0 - overlap


def compute_pair_tradeoff(
    graph: nx.Graph,
    source: NodeId,
    target: NodeId,
    risk_model: RiskModel,
    rain_key: str,
    min_delta_time_rel: float,
    min_delta_hazard_rel: float,
    min_divergence: float,
) -> PairTradeoffDetail | None:
    time_path = nx.shortest_path(graph, source, target, weight=lambda u, v, data: risk_model.edge_time_cost(data))
    hazard_path = nx.shortest_path(
        graph,
        source,
        target,
        weight=lambda u, v, data: risk_model.edge_hazard_cost(data, rain_key),
    )

    time_edges = path_edges(time_path)
    hazard_edges = path_edges(hazard_path)

    time_path_time = path_cost(graph, time_path, lambda data: risk_model.edge_time_cost(data))
    hazard_path_time = path_cost(graph, hazard_path, lambda data: risk_model.edge_time_cost(data))
    time_path_hazard = path_cost(graph, time_path, lambda data: risk_model.edge_hazard_cost(data, rain_key))
    hazard_path_hazard = path_cost(graph, hazard_path, lambda data: risk_model.edge_hazard_cost(data, rain_key))

    if time_path_time <= 0 or time_path_hazard <= 0:
        return None

    delta_time_rel = max(hazard_path_time - time_path_time, 0.0) / max(time_path_time, 1e-9)
    delta_hazard_rel = max(time_path_hazard - hazard_path_hazard, 0.0) / max(time_path_hazard, 1e-9)
    divergence = edge_divergence(time_edges, hazard_edges)

    if (
        delta_time_rel < min_delta_time_rel
        or delta_hazard_rel < min_delta_hazard_rel
        or divergence < min_divergence
    ):
        score = 0.0
    else:
        score = delta_time_rel * delta_hazard_rel * divergence

    return PairTradeoffDetail(
        rain_key=rain_key,
        time_path_time=time_path_time,
        hazard_path_time=hazard_path_time,
        time_path_hazard=time_path_hazard,
        hazard_path_hazard=hazard_path_hazard,
        delta_time_rel=delta_time_rel,
        delta_hazard_rel=delta_hazard_rel,
        divergence=divergence,
        score=score,
        union_edges=time_edges | hazard_edges,
        time_edges=time_edges,
        hazard_edges=hazard_edges,
    )


def sample_tradeoff_pairs(
    graph: nx.Graph,
    risk_model: RiskModel,
    rain_keys: Sequence[str],
    num_pairs: int,
    seed: int,
    min_time_minutes: float,
    max_time_minutes: float | None,
    min_delta_time_rel: float,
    min_delta_hazard_rel: float,
    min_divergence: float,
) -> List[PairTradeoffAggregate]:
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        return []

    sampled: List[PairTradeoffAggregate] = []
    seen_pairs: Set[Tuple[NodeId, NodeId]] = set()
    attempts = 0
    max_attempts = max(num_pairs * 30, 2000)

    while len(sampled) < num_pairs and attempts < max_attempts:
        attempts += 1
        source, target = rng.sample(nodes, 2)
        pair_key = canonical_edge(source, target)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        try:
            time_path = nx.shortest_path(graph, source, target, weight=lambda u, v, data: risk_model.edge_time_cost(data))
        except nx.NetworkXNoPath:
            continue

        time_cost = path_cost(graph, time_path, lambda data: risk_model.edge_time_cost(data))
        if time_cost < min_time_minutes:
            continue
        if max_time_minutes is not None and time_cost > max_time_minutes:
            continue

        details: List[PairTradeoffDetail] = []
        union_edges: Set[EdgeKey] = set()
        for rain_key in rain_keys:
            try:
                detail = compute_pair_tradeoff(
                    graph=graph,
                    source=source,
                    target=target,
                    risk_model=risk_model,
                    rain_key=rain_key,
                    min_delta_time_rel=min_delta_time_rel,
                    min_delta_hazard_rel=min_delta_hazard_rel,
                    min_divergence=min_divergence,
                )
            except nx.NetworkXNoPath:
                detail = None

            if detail is None:
                continue

            details.append(detail)
            union_edges |= detail.union_edges

        if not details:
            continue

        mean_score = sum(detail.score for detail in details) / len(details)
        sampled.append(
            PairTradeoffAggregate(
                source=source,
                target=target,
                mean_score=mean_score,
                max_score=max(detail.score for detail in details),
                mean_delta_time_rel=sum(detail.delta_time_rel for detail in details) / len(details),
                mean_delta_hazard_rel=sum(detail.delta_hazard_rel for detail in details) / len(details),
                mean_divergence=sum(detail.divergence for detail in details) / len(details),
                union_edges=union_edges,
                details=details,
            )
        )

    sampled.sort(key=lambda item: item.mean_score, reverse=True)
    return sampled

