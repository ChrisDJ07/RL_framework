from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>GraphML Web Viewer</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #f4f7fb; color: #18212b; }}
    .shell {{ display: grid; grid-template-columns: 320px 1fr; height: 100vh; }}
    .sidebar {{ padding: 16px; background: #ffffff; border-right: 1px solid #d6deea; overflow: auto; }}
    .canvas-wrap {{ position: relative; }}
    canvas {{ display: block; width: 100%; height: 100vh; background: #fbfdff; }}
    input, button {{ font: inherit; }}
    input {{ width: 100%; box-sizing: border-box; padding: 8px; margin: 6px 0; }}
    button {{ padding: 8px 10px; margin: 4px 4px 0 0; border: 1px solid #b9c6d8; background: #eef4fb; cursor: pointer; }}
    button:hover {{ background: #e1ecfa; }}
    .mono {{ font-family: Consolas, monospace; word-break: break-all; }}
    .section {{ margin-bottom: 18px; }}
    .small {{ color: #4f5f71; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="sidebar">
      <div class="section">
        <h2 style="margin:0 0 8px 0;">GraphML Viewer</h2>
        <div class="small">Nodes: <span id="nodeCount"></span></div>
        <div class="small">Edges: <span id="edgeCount"></span></div>
      </div>

      <div class="section">
        <label for="searchBox">Find node id</label>
        <input id="searchBox" type="text" placeholder="Enter node id" />
        <div>
          <button id="findBtn">Find</button>
          <button id="fitBtn">Fit</button>
        </div>
      </div>

      <div class="section">
        <h3 style="margin:0 0 8px 0;">Selection</h3>
        <div class="small mono" id="selectedNode">No node selected</div>
        <div class="small" id="selectedMeta"></div>
      </div>

      <div class="section">
        <h3 style="margin:0 0 8px 0;">Controls</h3>
        <div class="small">Left click: select nearest node</div>
        <div class="small">Drag: pan</div>
        <div class="small">Mouse wheel: zoom</div>
      </div>
    </div>
    <div class="canvas-wrap">
      <canvas id="graphCanvas"></canvas>
    </div>
  </div>

  <script>
    const graphData = __GRAPH_DATA__;
    const canvas = document.getElementById("graphCanvas");
    const ctx = canvas.getContext("2d");
    const searchBox = document.getElementById("searchBox");
    const selectedNodeEl = document.getElementById("selectedNode");
    const selectedMetaEl = document.getElementById("selectedMeta");
    document.getElementById("nodeCount").textContent = graphData.nodes.length;
    document.getElementById("edgeCount").textContent = graphData.edges.length;

    let scale = 1;
    let offsetX = 0;
    let offsetY = 0;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let selectedNodeId = null;

    const minX = Math.min(...graphData.nodes.map(n => n.x));
    const maxX = Math.max(...graphData.nodes.map(n => n.x));
    const minY = Math.min(...graphData.nodes.map(n => n.y));
    const maxY = Math.max(...graphData.nodes.map(n => n.y));

    const nodeMap = new Map(graphData.nodes.map(n => [n.id, n]));

    function resizeCanvas() {{
      canvas.width = canvas.clientWidth * window.devicePixelRatio;
      canvas.height = canvas.clientHeight * window.devicePixelRatio;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      draw();
    }}

    function fitView() {{
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const worldW = Math.max(maxX - minX, 1e-9);
      const worldH = Math.max(maxY - minY, 1e-9);
      const margin = 30;
      scale = Math.min((width - 2 * margin) / worldW, (height - 2 * margin) / worldH);
      offsetX = margin - minX * scale;
      offsetY = margin + maxY * scale;
      draw();
    }}

    function worldToScreen(x, y) {{
      return [x * scale + offsetX, offsetY - y * scale];
    }}

    function screenToWorld(sx, sy) {{
      return [(sx - offsetX) / scale, (offsetY - sy) / scale];
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);

      ctx.strokeStyle = "#a8b0ba";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (const edge of graphData.edges) {{
        const u = nodeMap.get(edge.u);
        const v = nodeMap.get(edge.v);
        if (!u || !v) continue;
        const [x1, y1] = worldToScreen(u.x, u.y);
        const [x2, y2] = worldToScreen(v.x, v.y);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }}
      ctx.stroke();

      for (const node of graphData.nodes) {{
        const [sx, sy] = worldToScreen(node.x, node.y);
        const selected = node.id === selectedNodeId;
        ctx.beginPath();
        ctx.fillStyle = selected ? "#d62828" : "#1d4ed8";
        ctx.arc(sx, sy, selected ? 4 : 2, 0, Math.PI * 2);
        ctx.fill();
      }}

      if (selectedNodeId && nodeMap.has(selectedNodeId)) {{
        const node = nodeMap.get(selectedNodeId);
        const [sx, sy] = worldToScreen(node.x, node.y);
        ctx.fillStyle = "#111827";
        ctx.font = "bold 12px Segoe UI";
        ctx.fillText(node.id, sx + 8, sy - 8);
      }}
    }}

    function selectNearestNode(sx, sy) {{
      let best = null;
      let bestDist2 = Infinity;
      for (const node of graphData.nodes) {{
        const [nx, ny] = worldToScreen(node.x, node.y);
        const dist2 = (nx - sx) ** 2 + (ny - sy) ** 2;
        if (dist2 < bestDist2) {{
          bestDist2 = dist2;
          best = node;
        }}
      }}
      if (!best || bestDist2 > 16 * 16) return;
      selectedNodeId = best.id;
      selectedNodeEl.textContent = best.id;
      selectedMetaEl.textContent = `x=${best.x}, y=${best.y}, degree=${best.degree}`;
      draw();
    }}

    function centerOnNode(nodeId) {{
      const node = nodeMap.get(nodeId);
      if (!node) return false;
      offsetX = canvas.clientWidth / 2 - node.x * scale;
      offsetY = canvas.clientHeight / 2 + node.y * scale;
      selectedNodeId = node.id;
      selectedNodeEl.textContent = node.id;
      selectedMetaEl.textContent = `x=${node.x}, y=${node.y}, degree=${node.degree}`;
      draw();
      return true;
    }}

    canvas.addEventListener("mousedown", event => {{
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
    }});

    canvas.addEventListener("mousemove", event => {{
      if (!dragging) return;
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      offsetX += dx;
      offsetY += dy;
      lastX = event.clientX;
      lastY = event.clientY;
      draw();
    }});

    canvas.addEventListener("mouseup", event => {{
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      dragging = false;
      if (Math.abs(dx) < 2 && Math.abs(dy) < 2) {{
        const rect = canvas.getBoundingClientRect();
        selectNearestNode(event.clientX - rect.left, event.clientY - rect.top);
      }}
    }});

    canvas.addEventListener("mouseleave", () => {{
      dragging = false;
    }});

    canvas.addEventListener("wheel", event => {{
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const sx = event.clientX - rect.left;
      const sy = event.clientY - rect.top;
      const [wxBefore, wyBefore] = screenToWorld(sx, sy);
      const factor = event.deltaY < 0 ? 1.1 : 0.9;
      scale *= factor;
      const [wxAfter, wyAfter] = screenToWorld(sx, sy);
      offsetX += (wxAfter - wxBefore) * scale;
      offsetY -= (wyAfter - wyBefore) * scale;
      draw();
    }}, {{ passive: false }});

    document.getElementById("findBtn").addEventListener("click", () => {{
      const ok = centerOnNode(searchBox.value.trim());
      if (!ok) {{
        selectedNodeEl.textContent = "Node not found";
        selectedMetaEl.textContent = "";
      }}
    }});

    searchBox.addEventListener("keydown", event => {{
      if (event.key === "Enter") {{
        document.getElementById("findBtn").click();
      }}
    }});

    document.getElementById("fitBtn").addEventListener("click", fitView);
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
    fitView();
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a standalone HTML GraphML viewer.")
    parser.add_argument("--graph", required=True, help="Input GraphML file.")
    parser.add_argument("--output", required=True, help="Output HTML file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    graph = nx.read_graphml(args.graph)
    payload = build_payload(graph)

    html = HTML_TEMPLATE.replace("__GRAPH_DATA__", json.dumps(payload))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Saved {output_path}")


def build_payload(graph: nx.Graph) -> dict[str, object]:
    fallback_nodes: list[str] = []
    positions: dict[str, tuple[float, float]] = {}
    for node_id, attrs in graph.nodes(data=True):
        try:
            positions[node_id] = (float(attrs["x"]), float(attrs["y"]))
        except (KeyError, TypeError, ValueError):
            fallback_nodes.append(node_id)

    if fallback_nodes:
        layout = nx.spring_layout(graph, seed=42)
        for node_id in fallback_nodes:
            x, y = layout[node_id]
            positions[node_id] = (float(x), float(y))

    nodes = [
        {
            "id": str(node_id),
            "x": positions[node_id][0],
            "y": positions[node_id][1],
            "degree": int(graph.degree[node_id]),
        }
        for node_id in graph.nodes()
    ]
    edges = [{"u": str(u), "v": str(v)} for u, v in graph.edges()]
    return {"nodes": nodes, "edges": edges}


if __name__ == "__main__":
    main()
