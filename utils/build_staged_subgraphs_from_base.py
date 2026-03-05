"""
Build nested connected staged subgraphs from a base GraphML.

This script uses only Python stdlib XML parsing so it works even if
networkx is not installed in the runtime environment.
"""

from __future__ import annotations

import argparse
import collections
import math
from pathlib import Path
import xml.etree.ElementTree as ET


GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
NS = {"g": GRAPHML_NS}


def _float_or_nan(value: str | None) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _build_key_name_map(root: ET.Element) -> dict[str, str]:
    key_name = {}
    for key_elem in root.findall("g:key", NS):
        key_id = key_elem.attrib.get("id")
        attr_name = key_elem.attrib.get("attr.name")
        if key_id and attr_name:
            key_name[key_id] = attr_name
    return key_name


def _find_graph(root: ET.Element) -> ET.Element:
    graph = root.find("g:graph", NS)
    if graph is None:
        raise RuntimeError("Invalid GraphML: missing <graph> element.")
    return graph


def _node_xy(node_elem: ET.Element, key_name_map: dict[str, str]) -> tuple[float, float]:
    x = float("nan")
    y = float("nan")
    for data_elem in node_elem.findall("g:data", NS):
        key = data_elem.attrib.get("key", "")
        name = key_name_map.get(key, "")
        if name == "x":
            x = _float_or_nan(data_elem.text)
        elif name == "y":
            y = _float_or_nan(data_elem.text)
    return x, y


def _choose_center_node(
    graph_elem: ET.Element,
    key_name_map: dict[str, str],
) -> str:
    coords = []
    for node_elem in graph_elem.findall("g:node", NS):
        node_id = node_elem.attrib["id"]
        x, y = _node_xy(node_elem, key_name_map)
        if math.isfinite(x) and math.isfinite(y):
            coords.append((node_id, x, y))

    if not coords:
        first_node = graph_elem.find("g:node", NS)
        if first_node is None:
            raise RuntimeError("Graph has no nodes.")
        return first_node.attrib["id"]

    mean_x = sum(x for _, x, _ in coords) / len(coords)
    mean_y = sum(y for _, _, y in coords) / len(coords)

    best_id = coords[0][0]
    best_d2 = float("inf")
    for node_id, x, y in coords:
        d2 = (x - mean_x) ** 2 + (y - mean_y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_id = node_id
    return best_id


def _build_adjacency(graph_elem: ET.Element) -> dict[str, set[str]]:
    node_ids = [n.attrib["id"] for n in graph_elem.findall("g:node", NS)]
    adj = {nid: set() for nid in node_ids}
    for edge_elem in graph_elem.findall("g:edge", NS):
        u = edge_elem.attrib.get("source")
        v = edge_elem.attrib.get("target")
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def _bfs_order(adj: dict[str, set[str]], start: str) -> list[str]:
    if start not in adj:
        raise RuntimeError(f"Start node {start} not found in graph.")

    order = []
    seen = {start}
    q = collections.deque([start])

    while q:
        u = q.popleft()
        order.append(u)
        for v in sorted(adj[u]):
            if v not in seen:
                seen.add(v)
                q.append(v)

    return order


def _prune_graph(
    graph_elem: ET.Element,
    keep_nodes: set[str],
) -> ET.Element:
    staged_graph = ET.fromstring(ET.tostring(graph_elem, encoding="utf-8"))

    for node_elem in list(staged_graph):
        if _local_name(node_elem.tag) != "node":
            continue
        if node_elem.attrib.get("id") not in keep_nodes:
            staged_graph.remove(node_elem)

    for edge_elem in list(staged_graph):
        if _local_name(edge_elem.tag) != "edge":
            continue
        u = edge_elem.attrib.get("source")
        v = edge_elem.attrib.get("target")
        if u not in keep_nodes or v not in keep_nodes:
            staged_graph.remove(edge_elem)

    return staged_graph


def build_stages(
    source_graphml: Path,
    output_dir: Path,
    sizes: list[int],
) -> None:
    tree = ET.parse(source_graphml)
    root = tree.getroot()
    graph_elem = _find_graph(root)
    key_name_map = _build_key_name_map(root)

    node_elems = graph_elem.findall("g:node", NS)
    total_nodes = len(node_elems)
    if total_nodes == 0:
        raise RuntimeError("Source graph has no nodes.")

    for n in sizes:
        if n <= 0:
            raise ValueError(f"Stage size must be positive, got {n}.")
        if n > total_nodes:
            raise ValueError(f"Stage size {n} exceeds source node count {total_nodes}.")

    center_node = _choose_center_node(graph_elem, key_name_map)
    adj = _build_adjacency(graph_elem)
    bfs_nodes = _bfs_order(adj, center_node)
    if len(bfs_nodes) < max(sizes):
        raise RuntimeError(
            f"BFS from center reached only {len(bfs_nodes)} nodes; "
            f"cannot build requested max size {max(sizes)}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {source_graphml}")
    print(f"Chosen center node: {center_node}")
    print(f"Reachable via BFS: {len(bfs_nodes)} nodes")

    for n in sorted(sizes, reverse=True):
        keep_nodes = set(bfs_nodes[:n])
        staged_graph = _prune_graph(graph_elem, keep_nodes)

        out_root = ET.fromstring(ET.tostring(root, encoding="utf-8"))
        out_graph = _find_graph(out_root)
        out_root.remove(out_graph)
        out_root.append(staged_graph)

        out_path = output_dir / f"selected_subgraph_n{n}.graphml"
        ET.ElementTree(out_root).write(out_path, encoding="utf-8", xml_declaration=True)

        edge_count = sum(1 for e in staged_graph if _local_name(e.tag) == "edge")
        print(f"Saved: {out_path} (nodes={n}, edges={edge_count})")


def _parse_sizes(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("No valid stage sizes provided.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nested staged GraphML files from a base GraphML.")
    parser.add_argument(
        "--source",
        type=str,
        default="data/subgraphs/selected_subgraph_n200.graphml",
        help="Path to base GraphML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/staged_subgraphs",
        help="Directory where staged GraphML files will be written.",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="200,150,100",
        help="Comma-separated stage sizes (largest to smallest recommended).",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source GraphML not found: {source}")

    sizes = _parse_sizes(args.sizes)
    build_stages(source_graphml=source, output_dir=Path(args.output_dir), sizes=sizes)


if __name__ == "__main__":
    main()


# python utils/build_staged_subgraphs_from_base.py --source data/subgraphs/selected_subgraph_n200.graphml --output-dir data/staged_subgraphs --sizes 200,150,100