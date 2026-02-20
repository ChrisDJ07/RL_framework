import argparse
import json
import math
import pickle
from pathlib import Path

import networkx as nx
import numpy as np


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


def convert(src_path, dst_path):
    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("rb") as f:
        G = pickle.load(f)

    H = G.__class__()
    H.graph.update({k: _to_graphml_value(v) for k, v in G.graph.items()})

    for n, d in G.nodes(data=True):
        H.add_node(n, **{k: _to_graphml_value(v) for k, v in d.items()})

    if G.is_multigraph():
        for u, v, k, d in G.edges(keys=True, data=True):
            H.add_edge(u, v, key=k, **{kk: _to_graphml_value(vv) for kk, vv in d.items()})
    else:
        for u, v, d in G.edges(data=True):
            H.add_edge(u, v, **{kk: _to_graphml_value(vv) for kk, vv in d.items()})

    nx.write_graphml(H, dst)
    return G, dst


def _describe(name, vals):
    if not vals:
        return f"{name}: missing"
    arr = np.asarray(vals, dtype=float)
    return (
        f"{name}: count={arr.size}, min={arr.min():.4f}, p25={np.percentile(arr,25):.4f}, "
        f"median={np.median(arr):.4f}, p75={np.percentile(arr,75):.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Convert hazard .gpickle graph into GraphML-safe file.")
    parser.add_argument("src", help="Path to input .gpickle")
    parser.add_argument("dst", help="Path to output .graphml")
    args = parser.parse_args()

    G, out = convert(args.src, args.dst)

    flood_vals = []
    land_vals = []
    comb_vals = []
    for _, _, data in G.edges(data=True):
        if "flood_hazard" in data:
            flood_vals.append(float(data["flood_hazard"]))
        if "landslide_hazard" in data:
            land_vals.append(float(data["landslide_hazard"]))
        if "combined_hazard" in data:
            comb_vals.append(float(data["combined_hazard"]))

    print(f"Wrote: {out}")
    print(f"Graph: {type(G).__name__}, nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    print(_describe("flood_hazard", flood_vals))
    print(_describe("landslide_hazard", land_vals))
    print(_describe("combined_hazard", comb_vals))


if __name__ == "__main__":
    main()
