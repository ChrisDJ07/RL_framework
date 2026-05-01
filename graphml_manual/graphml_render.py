from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import networkx as nx

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a GraphML file to PNG.")
    parser.add_argument("--graph", required=True, help="Input GraphML file.")
    parser.add_argument("--output", required=True, help="Output PNG file.")
    parser.add_argument("--figsize", type=float, nargs=2, default=(10.0, 10.0))
    parser.add_argument("--node-size", type=float, default=6.0)
    parser.add_argument("--edge-width", type=float, default=0.4)
    parser.add_argument(
        "--highlight-node",
        action="append",
        default=[],
        help="Optional node id to highlight. Repeat for multiple nodes.",
    )
    parser.add_argument(
        "--label-highlighted",
        action="store_true",
        help="Label only highlighted nodes.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    graph = nx.read_graphml(args.graph)
    positions = build_positions(graph)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    ax.set_aspect("equal")
    ax.axis("off")

    nx.draw_networkx_edges(
        graph,
        pos=positions,
        ax=ax,
        width=args.edge_width,
        edge_color="#9aa3ad",
        alpha=0.7,
    )
    nx.draw_networkx_nodes(
        graph,
        pos=positions,
        ax=ax,
        node_size=args.node_size,
        node_color="#1f78b4",
        linewidths=0,
    )

    highlighted = [node for node in args.highlight_node if node in graph]
    if highlighted:
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=highlighted,
            ax=ax,
            node_size=max(args.node_size * 5.0, 20.0),
            node_color="#e31a1c",
            linewidths=0,
        )
        if args.label_highlighted:
            labels = {node: node for node in highlighted}
            nx.draw_networkx_labels(graph, pos=positions, labels=labels, ax=ax, font_size=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def build_positions(graph: nx.Graph) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    missing = []
    for node_id, attrs in graph.nodes(data=True):
        try:
            x = float(attrs["x"])
            y = float(attrs["y"])
        except (KeyError, TypeError, ValueError):
            missing.append(node_id)
            continue
        positions[node_id] = (x, y)

    if missing:
        fallback = nx.spring_layout(graph, seed=42)
        for node_id in missing:
            positions[node_id] = tuple(fallback[node_id])
    return positions


if __name__ == "__main__":
    main()
