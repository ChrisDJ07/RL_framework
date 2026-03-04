"""
Select and export a hazard-diverse, redundant, spatially-spread subgraph from a larger OSM graph.
The selection process samples candidate subgraphs via BFS from random seed nodes, 
evaluates them based on a composite score of hazard distribution, redundancy, and spatial spread,
and exports the best candidate as GraphML along with a visualization.
"""

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm

try:
    import osmnx as ox
except ImportError:
    ox = None

try:
    from shapely import wkt as shapely_wkt
except ImportError:
    shapely_wkt = None

try:
    from shapely.geometry import MultiPoint
except ImportError:
    MultiPoint = None

try:
    from utils.graph_utils import get_raw_osm_graph
except ImportError:
    from graph_utils import get_raw_osm_graph


DEFAULT_PREBUILT = "data/la_trinidad_hazard_graph.graphml"

HAZARD_BINS = ("safe", "very_low", "low", "moderate", "high", "very_high")
TARGET_HAZARD_DIST = {
    "safe": 0.15,
    "very_low": 0.40,
    "low": 0.20,
    "moderate": 0.15,
    "high": 0.05,
    "very_high": 0.05,
}


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_undirected_simple(raw_graph):
    G_u = nx.Graph()
    for n, d in raw_graph.nodes(data=True):
        G_u.add_node(n, **dict(d))

    if raw_graph.is_multigraph():
        for u, v, _, d in raw_graph.edges(keys=True, data=True):
            payload = dict(d)
            cand_len = _safe_float(payload.get("length", float("inf")), float("inf"))
            if G_u.has_edge(u, v):
                prev_len = _safe_float(G_u[u][v].get("length", float("inf")), float("inf"))
                if cand_len >= prev_len:
                    continue
            G_u.add_edge(u, v, **payload)
    else:
        for u, v, d in raw_graph.edges(data=True):
            payload = dict(d)
            cand_len = _safe_float(payload.get("length", float("inf")), float("inf"))
            if G_u.has_edge(u, v):
                prev_len = _safe_float(G_u[u][v].get("length", float("inf")), float("inf"))
                if cand_len >= prev_len:
                    continue
            G_u.add_edge(u, v, **payload)

    return G_u


def _load_source_graph(prebuilt_graphml, force_download, min_nodes):
    if prebuilt_graphml:
        G_raw = nx.read_graphml(prebuilt_graphml)
    else:
        G_raw = get_raw_osm_graph(min_nodes=min_nodes, force_download=force_download)

    if ox is not None:
        try:
            if hasattr(ox, "convert") and hasattr(ox.convert, "to_undirected"):
                G_work = ox.convert.to_undirected(G_raw)
            else:
                G_work = ox.utils_graph.get_undirected(G_raw)
        except Exception:
            G_work = _to_undirected_simple(G_raw)
    else:
        G_work = _to_undirected_simple(G_raw)

    largest_cc = max(nx.connected_components(G_work), key=len)
    G_work = G_work.subgraph(largest_cc).copy()
    return G_work


def _edge_overall_hazard(data, flood_attr, landslide_attr, combined_attr, flood_weight):
    if combined_attr:
        combined = _safe_float(data.get(combined_attr, np.nan), np.nan)
        if np.isfinite(combined):
            if combined > 1.0:
                combined = combined / 2.0
            return float(np.clip(combined, 0.0, 1.0))

    flood = _safe_float(data.get(flood_attr, 0.0), 0.0)
    landslide = _safe_float(data.get(landslide_attr, 0.0), 0.0)
    return float(np.clip(flood_weight * flood + (1.0 - flood_weight) * landslide, 0.0, 1.0))


def _hazard_bucket(value):
    v = float(np.clip(value, 0.0, 1.0))
    if v <= 1e-12:
        return "safe"
    if v <= 0.2:
        return "very_low"
    if v <= 0.4:
        return "low"
    if v <= 0.6:
        return "moderate"
    if v <= 0.8:
        return "high"
    return "very_high"


def _node_lon_lat(data):
    lon = _safe_float(data.get("x", data.get("lon", data.get("longitude", np.nan))), np.nan)
    lat = _safe_float(data.get("y", data.get("lat", data.get("latitude", np.nan))), np.nan)
    if np.isfinite(lon) and np.isfinite(lat):
        return float(lon), float(lat)
    return None


def _compute_area_km2(graph):
    pts = []
    for _, d in graph.nodes(data=True):
        lonlat = _node_lon_lat(d)
        if lonlat is not None:
            pts.append(lonlat)
    if len(pts) < 3:
        return 0.0

    pts = np.unique(np.asarray(pts, dtype=float), axis=0)
    if pts.shape[0] < 3:
        return 0.0

    lat0 = math.radians(float(np.mean(pts[:, 1])))
    r = 6_371_000.0
    x = np.radians(pts[:, 0]) * r * math.cos(lat0)
    y = np.radians(pts[:, 1]) * r
    xy = np.column_stack([x, y])

    if MultiPoint is not None:
        area_m2 = float(MultiPoint(xy).convex_hull.area)
        return area_m2 / 1_000_000.0

    # Fallback: bounding-box area in projected space.
    width = float(np.max(x) - np.min(x))
    height = float(np.max(y) - np.min(y))
    return max(width * height, 0.0) / 1_000_000.0


def _extract_candidate_subgraph(graph, seed_node, num_nodes):
    bfs_nodes = list(nx.bfs_tree(graph, seed_node).nodes())
    if len(bfs_nodes) < num_nodes:
        return None
    picked = bfs_nodes[:num_nodes]
    sub = graph.subgraph(picked).copy()
    if not nx.is_connected(sub):
        return None
    return sub


def _evaluate_candidate(sub, args):
    n = sub.number_of_nodes()
    m = sub.number_of_edges()
    if n <= 1 or m == 0:
        return None

    hazard_vals = []
    for _, _, d in sub.edges(data=True):
        hazard_vals.append(
            _edge_overall_hazard(
                d,
                flood_attr=args.flood_attr,
                landslide_attr=args.landslide_attr,
                combined_attr=args.combined_hazard_attr,
                flood_weight=args.flood_weight,
            )
        )

    counts = {k: 0 for k in HAZARD_BINS}
    for h in hazard_vals:
        counts[_hazard_bucket(h)] += 1
    total_edges = float(len(hazard_vals))
    observed = {k: counts[k] / total_edges for k in HAZARD_BINS}

    l1 = sum(abs(observed[k] - TARGET_HAZARD_DIST[k]) for k in HAZARD_BINS)
    hazard_dist_score = max(0.0, 1.0 - 0.5 * l1)
    tail_ratio = observed["high"] + observed["very_high"]
    tail_score = min(tail_ratio / max(args.min_high_hazard_ratio, 1e-9), 1.0)
    hazard_score = 0.75 * hazard_dist_score + 0.25 * tail_score

    bridges = list(nx.bridges(sub))
    bridge_ratio = len(bridges) / max(m, 1)
    high_bridge_count = 0
    for u, v in bridges:
        edge_data = sub[u][v]
        h = _edge_overall_hazard(
            edge_data,
            flood_attr=args.flood_attr,
            landslide_attr=args.landslide_attr,
            combined_attr=args.combined_hazard_attr,
            flood_weight=args.flood_weight,
        )
        if h >= args.high_hazard_bridge_threshold:
            high_bridge_count += 1
    high_bridge_ratio = high_bridge_count / max(len(bridges), 1)

    cycle_rank = max(m - n + 1, 0)
    cycle_density = cycle_rank / max(n, 1)
    bridge_component = 1.0 - min(bridge_ratio / max(args.max_bridge_ratio, 1e-9), 1.0)
    high_bridge_component = 1.0 - min(high_bridge_ratio / max(args.max_high_bridge_ratio, 1e-9), 1.0)
    cycle_component = min(cycle_density / max(args.min_cycle_density, 1e-9), 1.0)
    redundancy_score = 0.50 * bridge_component + 0.30 * high_bridge_component + 0.20 * cycle_component

    area_km2 = _compute_area_km2(sub)
    area_per_node = area_km2 / max(n, 1)
    density = nx.density(sub)
    area_component = min(area_per_node / max(args.min_area_per_node_km2, 1e-9), 1.0)
    density_component = 1.0 - min(density / max(args.max_density, 1e-9), 1.0)
    spatial_score = 0.70 * area_component + 0.30 * density_component

    weight_sum = args.w_hazard + args.w_redundancy + args.w_spatial
    total_score = (
        args.w_hazard * hazard_score
        + args.w_redundancy * redundancy_score
        + args.w_spatial * spatial_score
    ) / max(weight_sum, 1e-9)

    return {
        "graph": sub,
        "score_total": float(total_score),
        "score_hazard": float(hazard_score),
        "score_redundancy": float(redundancy_score),
        "score_spatial": float(spatial_score),
        "hazard_dist_score": float(hazard_dist_score),
        "tail_ratio": float(tail_ratio),
        "bridge_ratio": float(bridge_ratio),
        "high_bridge_ratio": float(high_bridge_ratio),
        "cycle_density": float(cycle_density),
        "density": float(density),
        "area_km2": float(area_km2),
        "area_per_node_km2": float(area_per_node),
        "n": n,
        "m": m,
        "hazard_distribution": observed,
        "bridge_count": int(len(bridges)),
        "high_bridge_count": int(high_bridge_count),
    }


def _convert_wkt_geometries_for_plot(graph):
    if shapely_wkt is None:
        return
    if graph.is_multigraph():
        edge_iter = graph.edges(keys=True, data=True)
        for _, _, _, data in edge_iter:
            geom = data.get("geometry")
            if isinstance(geom, str):
                try:
                    data["geometry"] = shapely_wkt.loads(geom)
                except Exception:
                    data.pop("geometry", None)
    else:
        edge_iter = graph.edges(data=True)
        for _, _, data in edge_iter:
            geom = data.get("geometry")
            if isinstance(geom, str):
                try:
                    data["geometry"] = shapely_wkt.loads(geom)
                except Exception:
                    data.pop("geometry", None)


def _visualize_subgraph(sub, metrics, args, save_path, show_plot=False):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer a map-like view when OSMnx is available and coordinates exist.
    has_xy = all(_node_lon_lat(d) is not None for _, d in sub.nodes(data=True))
    if ox is not None and has_xy:
        g_plot = sub.copy()
        _convert_wkt_geometries_for_plot(g_plot)
        try:
            fig, ax = ox.plot_graph(
                g_plot,
                node_size=14,
                edge_linewidth=1.0,
                edge_color="#2b2b2b",
                bgcolor="white",
                show=False,
                close=False,
            )
            ax.set_title(
                f"Selected Subgraph | Nodes={metrics['n']} Edges={metrics['m']} "
                f"Score={metrics['score_total']:.3f}"
            )
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
            return
        except Exception:
            pass

    pos = {}
    for n, d in sub.nodes(data=True):
        lonlat = _node_lon_lat(d)
        if lonlat is not None:
            pos[n] = lonlat
    if len(pos) != sub.number_of_nodes():
        pos = nx.spring_layout(sub, seed=args.seed)

    bridges = set(tuple(sorted(e)) for e in nx.bridges(sub))
    hazards = []
    for u, v, d in sub.edges(data=True):
        h = _edge_overall_hazard(
            d,
            flood_attr=args.flood_attr,
            landslide_attr=args.landslide_attr,
            combined_attr=args.combined_hazard_attr,
            flood_weight=args.flood_weight,
        )
        hazards.append((u, v, h))

    fig, ax = plt.subplots(figsize=(10, 8))
    non_bridge_edges = []
    non_bridge_colors = []
    bridge_edges = []

    for u, v, h in hazards:
        key = tuple(sorted((u, v)))
        if key in bridges:
            bridge_edges.append((u, v))
        else:
            non_bridge_edges.append((u, v))
            non_bridge_colors.append(h)

    if non_bridge_edges:
        nx.draw_networkx_edges(
            sub,
            pos,
            ax=ax,
            edgelist=non_bridge_edges,
            edge_color=non_bridge_colors,
            edge_cmap=cm.viridis,
            edge_vmin=0.0,
            edge_vmax=1.0,
            width=1.4,
            alpha=0.95,
        )
    if bridge_edges:
        nx.draw_networkx_edges(
            sub,
            pos,
            ax=ax,
            edgelist=bridge_edges,
            edge_color="black",
            width=2.3,
            alpha=0.95,
        )

    nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=24, node_color="#f3f4f6", edgecolors="#4b5563", linewidths=0.5)

    sm = cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
    cbar.set_label("Overall Hazard", rotation=90)

    ax.set_title(
        f"Selected Subgraph | Nodes={metrics['n']} Edges={metrics['m']} "
        f"Score={metrics['score_total']:.3f}\n"
        f"Hazard={metrics['score_hazard']:.3f} Redundancy={metrics['score_redundancy']:.3f} "
        f"Spatial={metrics['score_spatial']:.3f} | Bridges={metrics['bridge_count']}"
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _to_graphml_value(v):
    if v is None:
        return ""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, (bool, int, float, str)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return ""
        return v
    if hasattr(v, "wkt"):
        return v.wkt
    if isinstance(v, (list, tuple, set, dict)):
        try:
            return json.dumps(v)
        except Exception:
            return str(v)
    return str(v)


def _to_graphml_safe_graph(G):
    H = G.__class__()
    H.graph.update({k: _to_graphml_value(v) for k, v in G.graph.items()})

    for n, d in G.nodes(data=True):
        payload = {k: _to_graphml_value(v) for k, v in d.items()}
        H.add_node(n, **payload)

    if G.is_multigraph():
        for u, v, k, d in G.edges(keys=True, data=True):
            payload = {kk: _to_graphml_value(vv) for kk, vv in d.items()}
            H.add_edge(u, v, key=k, **payload)
    else:
        for u, v, d in G.edges(data=True):
            payload = {kk: _to_graphml_value(vv) for kk, vv in d.items()}
            H.add_edge(u, v, **payload)
    return H


def _print_top(candidates, top_k):
    k = min(top_k, len(candidates))
    print(f"Top {k} candidate(s):")
    for i in range(k):
        c = candidates[i]
        dist = c["hazard_distribution"]
        print(
            f"[{i + 1}] score={c['score_total']:.4f} | nodes={c['n']} edges={c['m']} "
            f"| hazard={c['score_hazard']:.3f} red={c['score_redundancy']:.3f} spatial={c['score_spatial']:.3f} "
            f"| bridges={c['bridge_count']} (high={c['high_bridge_count']}) "
            f"| area={c['area_km2']:.3f} km^2 density={c['density']:.4f}"
        )
        print(
            "    hazard_dist="
            + ", ".join([f"{k_}:{dist[k_]:.2%}" for k_ in HAZARD_BINS])
        )


def select_best_subgraph(args):
    source_graph = _load_source_graph(
        prebuilt_graphml=args.prebuilt_graphml,
        force_download=args.force_download,
        min_nodes=args.num_nodes,
    )
    if source_graph.number_of_nodes() < args.num_nodes:
        raise RuntimeError(
            f"Source graph has only {source_graph.number_of_nodes()} nodes; cannot select {args.num_nodes}-node subgraph."
        )

    rng = random.Random(args.seed)
    nodes = list(source_graph.nodes())
    seen = set()
    candidates = []
    attempts = 0
    max_attempts = max(args.num_candidates * 30, 500)

    while len(candidates) < args.num_candidates and attempts < max_attempts:
        attempts += 1
        seed = rng.choice(nodes)
        sub = _extract_candidate_subgraph(source_graph, seed, args.num_nodes)
        if sub is None:
            continue
        signature = frozenset(sub.nodes())
        if signature in seen:
            continue
        seen.add(signature)
        metrics = _evaluate_candidate(sub, args)
        if metrics is None:
            continue
        metrics["seed"] = seed
        candidates.append(metrics)

    if not candidates:
        raise RuntimeError("No valid candidate subgraph was generated.")

    candidates.sort(key=lambda x: x["score_total"], reverse=True)
    return source_graph, candidates


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Select and export a hazard-diverse, redundant, spatially-spread subgraph."
    )
    parser.add_argument("--num-nodes", type=int, required=True, help="Target node count for selected subgraph.")
    parser.add_argument("--num-candidates", type=int, default=250, help="Number of unique candidates to score.")
    parser.add_argument("--seed", type=int, default=40, help="Random seed for candidate sampling.")
    parser.add_argument("--prebuilt-graphml", type=str, default=DEFAULT_PREBUILT, help="Input source graph GraphML.")
    parser.add_argument("--force-download", action="store_true", help="Download source graph via OSMnx when not using prebuilt GraphML.")

    parser.add_argument("--flood-attr", type=str, default="flood_hazard")
    parser.add_argument("--landslide-attr", type=str, default="landslide_hazard")
    parser.add_argument("--combined-hazard-attr", type=str, default="combined_hazard")
    parser.add_argument("--flood-weight", type=float, default=0.5, help="Flood weight when combined hazard is unavailable.")

    parser.add_argument("--min-high-hazard-ratio", type=float, default=0.05)
    parser.add_argument("--high-hazard-bridge-threshold", type=float, default=0.6)
    parser.add_argument("--max-bridge-ratio", type=float, default=0.25)
    parser.add_argument("--max-high-bridge-ratio", type=float, default=0.20)
    parser.add_argument("--min-cycle-density", type=float, default=0.03)
    parser.add_argument("--min-area-per-node-km2", type=float, default=0.003)
    parser.add_argument("--max-density", type=float, default=0.08)

    parser.add_argument("--w-hazard", type=float, default=0.45)
    parser.add_argument("--w-redundancy", type=float, default=0.35)
    parser.add_argument("--w-spatial", type=float, default=0.20)

    parser.add_argument("--top-k", type=int, default=5, help="Print top-K scored candidates.")
    parser.add_argument("--output-graphml", type=str, default="", help="Output GraphML path. Defaults to data/subgraphs/selected_subgraph_n{N}.graphml.")
    parser.add_argument("--output-plot", type=str, default="", help="Output plot path. Defaults to results/subgraphs/selected_subgraph_n{N}.png.")
    parser.add_argument("--show-plot", action="store_true", help="Show matplotlib plot window.")
    parser.add_argument("--dry-run", action="store_true", help="Score and print candidates without writing files.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    source_graph, candidates = select_best_subgraph(args)
    _print_top(candidates, args.top_k)

    best = candidates[0]
    best_graph = best["graph"].copy()
    if not best_graph.graph.get("crs"):
        best_graph.graph["crs"] = source_graph.graph.get("crs", "epsg:4326")
    best_graph.graph["selection_score_total"] = float(best["score_total"])
    best_graph.graph["selection_score_hazard"] = float(best["score_hazard"])
    best_graph.graph["selection_score_redundancy"] = float(best["score_redundancy"])
    best_graph.graph["selection_score_spatial"] = float(best["score_spatial"])
    best_graph.graph["selection_seed_node"] = str(best["seed"])
    best_graph.graph["selection_hazard_distribution"] = json.dumps(best["hazard_distribution"])

    if args.output_graphml:
        output_graphml = Path(args.output_graphml)
    else:
        output_graphml = Path("data/subgraphs/") / f"selected_subgraph_n{args.num_nodes}.graphml"
    if args.output_plot:
        output_plot = Path(args.output_plot)
    else:
        output_plot = Path("results/subgraphs/") / f"selected_subgraph_n{args.num_nodes}.png"

    if args.dry_run:
        print("Dry run mode enabled. No files written.")
        return

    output_graphml.parent.mkdir(parents=True, exist_ok=True)
    graphml_safe = _to_graphml_safe_graph(best_graph)
    nx.write_graphml(graphml_safe, output_graphml)
    print(f"Saved selected subgraph GraphML to: {output_graphml}")

    _visualize_subgraph(best_graph, best, args, output_plot, show_plot=args.show_plot)
    print(f"Saved selected subgraph visualization to: {output_plot}")


if __name__ == "__main__":
    main()
