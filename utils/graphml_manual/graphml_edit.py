from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple GraphML node add/remove helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add-node", help="Add a node to a GraphML file.")
    add_common_graph_args(add_parser)
    add_parser.add_argument("--node-id", required=True, help="New node id.")
    add_parser.add_argument("--x", required=True, type=float, help="Node x coordinate.")
    add_parser.add_argument("--y", required=True, type=float, help="Node y coordinate.")
    add_parser.add_argument("--street-count", type=int, default=0, help="Optional street_count.")

    remove_parser = subparsers.add_parser("remove-node", help="Remove a node from a GraphML file.")
    add_common_graph_args(remove_parser)
    remove_parser.add_argument("--node-id", required=True, help="Existing node id to remove.")

    remove_many_parser = subparsers.add_parser(
        "remove-nodes",
        help="Remove multiple nodes from a GraphML file.",
    )
    add_common_graph_args(remove_many_parser)
    remove_many_parser.add_argument(
        "--node-id",
        action="append",
        default=[],
        help="Node id to remove. Repeat this flag for multiple nodes.",
    )
    remove_many_parser.add_argument(
        "--node-file",
        help="Optional text file containing one node id per line.",
    )

    import_parser = subparsers.add_parser(
        "import-nodes",
        help="Import existing nodes from a base graph, preserving node and edge attributes.",
    )
    add_common_graph_args(import_parser)
    import_parser.add_argument(
        "--base-graph",
        required=True,
        help="Base GraphML file to copy nodes and edges from.",
    )
    import_parser.add_argument(
        "--node-id",
        action="append",
        default=[],
        help="Node id to import. Repeat this flag for multiple nodes.",
    )
    import_parser.add_argument(
        "--node-file",
        help="Optional text file containing one node id per line.",
    )

    list_parser = subparsers.add_parser("show-node", help="Show one node's attributes.")
    list_parser.add_argument("--graph", required=True, help="Input GraphML file.")
    list_parser.add_argument("--node-id", required=True, help="Node id to inspect.")

    return parser


def add_common_graph_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph", required=True, help="Input GraphML file.")
    parser.add_argument("--output", required=True, help="Output GraphML file.")


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "show-node":
        graph = nx.read_graphml(args.graph)
        if args.node_id not in graph:
            raise SystemExit(f"Node {args.node_id!r} not found in {args.graph}.")
        print(f"Node {args.node_id}")
        for key, value in sorted(graph.nodes[args.node_id].items()):
            print(f"  {key}: {value}")
        return

    graph = nx.read_graphml(args.graph)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.command == "add-node":
        if args.node_id in graph:
            raise SystemExit(f"Node {args.node_id!r} already exists.")
        graph.add_node(
            args.node_id,
            x=float(args.x),
            y=float(args.y),
            street_count=int(args.street_count),
        )
        nx.write_graphml(graph, output_path)
        print(f"Added node {args.node_id!r} and wrote {output_path}")
        return

    if args.command == "remove-node":
        if args.node_id not in graph:
            raise SystemExit(f"Node {args.node_id!r} not found.")
        graph.remove_node(args.node_id)
        nx.write_graphml(graph, output_path)
        print(f"Removed node {args.node_id!r} and wrote {output_path}")
        return

    if args.command == "remove-nodes":
        node_ids = collect_node_ids(args.node_id, args.node_file)
        if not node_ids:
            raise SystemExit("No node ids provided. Use --node-id or --node-file.")

        missing = [node_id for node_id in node_ids if node_id not in graph]
        if missing:
            missing_str = ", ".join(missing)
            raise SystemExit(f"Node ids not found in target graph: {missing_str}")

        for node_id in node_ids:
            graph.remove_node(node_id)

        nx.write_graphml(graph, output_path)
        print(f"Removed {len(node_ids)} nodes and wrote {output_path}")
        return

    if args.command == "import-nodes":
        base_graph = nx.read_graphml(args.base_graph)
        node_ids = collect_node_ids(args.node_id, args.node_file)
        if not node_ids:
            raise SystemExit("No node ids provided. Use --node-id or --node-file.")

        imported_nodes, imported_edges = import_nodes_from_base(
            target_graph=graph,
            base_graph=base_graph,
            node_ids=node_ids,
        )
        nx.write_graphml(graph, output_path)
        print(
            f"Imported {imported_nodes} nodes and {imported_edges} edges from {args.base_graph} "
            f"into {output_path}"
        )
        return


def collect_node_ids(node_ids: list[str], node_file: str | None) -> list[str]:
    combined = list(node_ids)
    if node_file:
        file_path = Path(node_file)
        if not file_path.exists():
            raise SystemExit(f"Node file not found: {node_file}")
        file_ids = [
            line.strip()
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        combined.extend(file_ids)

    seen: set[str] = set()
    deduped: list[str] = []
    for node_id in combined:
        if node_id not in seen:
            seen.add(node_id)
            deduped.append(node_id)
    return deduped


def import_nodes_from_base(
    target_graph: nx.Graph,
    base_graph: nx.Graph,
    node_ids: list[str],
) -> tuple[int, int]:
    missing = [node_id for node_id in node_ids if node_id not in base_graph]
    if missing:
        missing_str = ", ".join(missing)
        raise SystemExit(f"Node ids not found in base graph: {missing_str}")

    imported_nodes = 0
    imported_edges = 0
    for node_id in node_ids:
        if node_id in target_graph:
            continue
        target_graph.add_node(node_id, **dict(base_graph.nodes[node_id]))
        imported_nodes += 1

    target_node_set = set(target_graph.nodes())
    for node_id in node_ids:
        for neighbor in base_graph.neighbors(node_id):
            if neighbor not in target_node_set:
                continue
            imported_edges += copy_edges_between(
                target_graph=target_graph,
                base_graph=base_graph,
                node_id=node_id,
                neighbor=neighbor,
            )

    return imported_nodes, imported_edges


def copy_edges_between(
    target_graph: nx.Graph,
    base_graph: nx.Graph,
    node_id: str,
    neighbor: str,
) -> int:
    copied = 0

    if base_graph.is_multigraph():
        edge_data = base_graph.get_edge_data(node_id, neighbor, default={})
        for key, attrs in edge_data.items():
            if target_graph.is_multigraph():
                if target_graph.has_edge(node_id, neighbor, key=key):
                    continue
                target_graph.add_edge(node_id, neighbor, key=key, **dict(attrs))
            else:
                if target_graph.has_edge(node_id, neighbor):
                    continue
                target_graph.add_edge(node_id, neighbor, **dict(attrs))
            copied += 1
        return copied

    attrs = base_graph.get_edge_data(node_id, neighbor, default=None)
    if attrs is None:
        return 0
    if target_graph.has_edge(node_id, neighbor):
        return 0
    target_graph.add_edge(node_id, neighbor, **dict(attrs))
    return 1


if __name__ == "__main__":
    main()
