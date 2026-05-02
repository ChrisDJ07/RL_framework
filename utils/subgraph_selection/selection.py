from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Hashable, Iterable, List, Mapping, Sequence, Set, Tuple

import networkx as nx

from utils.subgraph_selection.graph_io import EdgeKey, NodeId, canonical_edge
from utils.subgraph_selection.tradeoff import PairTradeoffAggregate


@dataclass
class CandidateSubgraph:
    seed: NodeId
    graph: nx.Graph
    score_total: float
    score_pair_mean: float
    score_pair_fraction: float
    score_edge_capture: float
    internal_pair_count: int
    weighted_edge_capture: float


def aggregate_corridor_weights(
    pair_results: Sequence[PairTradeoffAggregate],
    exclusive_edge_bonus: float,
) -> Tuple[Dict[EdgeKey, float], Dict[NodeId, float]]:
    edge_weights: Dict[EdgeKey, float] = {}
    node_weights: Dict[NodeId, float] = {}

    for pair in pair_results:
        for detail in pair.details:
            if detail.score <= 0:
                continue

            symmetric_edges = detail.time_edges ^ detail.hazard_edges
            for edge in detail.union_edges:
                edge_weights[edge] = edge_weights.get(edge, 0.0) + detail.score
            for edge in symmetric_edges:
                edge_weights[edge] = edge_weights.get(edge, 0.0) + detail.score * max(exclusive_edge_bonus - 1.0, 0.0)

    for (u, v), weight in edge_weights.items():
        node_weights[u] = node_weights.get(u, 0.0) + weight
        node_weights[v] = node_weights.get(v, 0.0) + weight

    return edge_weights, node_weights


def select_seed_nodes(
    graph: nx.Graph,
    node_weights: Mapping[NodeId, float],
    max_seeds: int,
) -> List[NodeId]:
    weighted_nodes = sorted(node_weights.items(), key=lambda item: item[1], reverse=True)
    if not weighted_nodes:
        weighted_nodes = sorted(graph.degree, key=lambda item: item[1], reverse=True)
        return [node for node, _ in weighted_nodes[:max_seeds]]
    return [node for node, _ in weighted_nodes[:max_seeds]]


def grow_tradeoff_subgraph(
    graph: nx.Graph,
    seed: NodeId,
    node_weights: Mapping[NodeId, float],
    edge_weights: Mapping[EdgeKey, float],
    num_nodes: int,
    distance_penalty: float,
    core_fraction: float,
) -> nx.Graph:
    hop_dist = nx.single_source_shortest_path_length(graph, seed)
    positive_nodes = [node for node, weight in node_weights.items() if weight > 0]
    ranked_nodes = sorted(
        positive_nodes,
        key=lambda node: node_weights.get(node, 0.0) / (1.0 + hop_dist.get(node, 10**9)),
        reverse=True,
    )

    selected: Set[NodeId] = {seed}
    core_target = min(max(5, int(num_nodes * core_fraction)), max(num_nodes - 1, 1))

    for node in ranked_nodes:
        if len(selected) >= max(int(num_nodes * 0.6), core_target):
            break
        try:
            path = nx.shortest_path(graph, seed, node)
        except nx.NetworkXNoPath:
            continue
        selected.update(path)
        if len(selected) >= num_nodes:
            break

    while len(selected) < num_nodes:
        frontier = {
            nbr
            for node in selected
            for nbr in graph.neighbors(node)
            if nbr not in selected
        }
        if not frontier:
            break

        best_node = max(
            frontier,
            key=lambda node: _frontier_score(
                node=node,
                selected=selected,
                graph=graph,
                node_weights=node_weights,
                edge_weights=edge_weights,
                hop_dist=hop_dist,
                distance_penalty=distance_penalty,
            ),
        )
        selected.add(best_node)

    subgraph = graph.subgraph(selected).copy()
    if subgraph.number_of_nodes() > num_nodes:
        subgraph = _trim_connected_subgraph(subgraph, node_weights, num_nodes)
    return subgraph


def _frontier_score(
    node: NodeId,
    selected: Set[NodeId],
    graph: nx.Graph,
    node_weights: Mapping[NodeId, float],
    edge_weights: Mapping[EdgeKey, float],
    hop_dist: Mapping[NodeId, int],
    distance_penalty: float,
) -> float:
    corridor_bonus = 0.0
    for neighbor in graph.neighbors(node):
        if neighbor in selected:
            corridor_bonus += edge_weights.get(canonical_edge(node, neighbor), 0.0)
    return (
        node_weights.get(node, 0.0)
        + 0.75 * corridor_bonus
        - distance_penalty * hop_dist.get(node, 0)
    )


def _trim_connected_subgraph(graph: nx.Graph, node_weights: Mapping[NodeId, float], target_nodes: int) -> nx.Graph:
    subgraph = graph.copy()
    while subgraph.number_of_nodes() > target_nodes:
        leaves = [node for node, degree in subgraph.degree() if degree <= 1]
        if not leaves:
            break
        candidate = min(leaves, key=lambda node: node_weights.get(node, 0.0))
        trial = subgraph.copy()
        trial.remove_node(candidate)
        if nx.is_connected(trial):
            subgraph = trial
        else:
            break
    return subgraph


def score_candidate_subgraph(
    subgraph: nx.Graph,
    pair_results: Sequence[PairTradeoffAggregate],
    edge_weights: Mapping[EdgeKey, float],
    interesting_threshold: float,
) -> CandidateSubgraph | None:
    node_set = set(subgraph.nodes())
    edge_set = {canonical_edge(u, v) for u, v in subgraph.edges()}
    internal_pairs = [pair for pair in pair_results if pair.source in node_set and pair.target in node_set]
    if not internal_pairs:
        return None

    weighted_scores: List[float] = []
    interesting = 0
    for pair in internal_pairs:
        if not pair.union_edges:
            continue
        edge_coverage = len(pair.union_edges & edge_set) / len(pair.union_edges)
        contribution = pair.mean_score * edge_coverage
        weighted_scores.append(contribution)
        if contribution >= interesting_threshold:
            interesting += 1

    if not weighted_scores:
        return None

    total_edge_weight = sum(edge_weights.values()) or 1e-9
    captured_edge_weight = sum(weight for edge, weight in edge_weights.items() if edge in edge_set)
    edge_capture = captured_edge_weight / total_edge_weight

    pair_mean = mean(weighted_scores)
    pair_fraction = interesting / len(weighted_scores)
    score_total = 0.65 * pair_mean + 0.20 * pair_fraction + 0.15 * edge_capture

    return CandidateSubgraph(
        seed="",
        graph=subgraph,
        score_total=score_total,
        score_pair_mean=pair_mean,
        score_pair_fraction=pair_fraction,
        score_edge_capture=edge_capture,
        internal_pair_count=len(weighted_scores),
        weighted_edge_capture=captured_edge_weight,
    )


def graph_positions(graph: nx.Graph) -> Dict[NodeId, Tuple[float, float]]:
    positions: Dict[NodeId, Tuple[float, float]] = {}
    for node, data in graph.nodes(data=True):
        try:
            positions[node] = (float(data["x"]), float(data["y"]))
        except (KeyError, TypeError, ValueError):
            continue
    return positions


def candidate_to_json(candidate: CandidateSubgraph) -> Dict[str, object]:
    return {
        "seed": candidate.seed,
        "score_total": candidate.score_total,
        "score_pair_mean": candidate.score_pair_mean,
        "score_pair_fraction": candidate.score_pair_fraction,
        "score_edge_capture": candidate.score_edge_capture,
        "internal_pair_count": candidate.internal_pair_count,
        "weighted_edge_capture": candidate.weighted_edge_capture,
        "num_nodes": candidate.graph.number_of_nodes(),
        "num_edges": candidate.graph.number_of_edges(),
    }

