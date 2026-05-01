from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import networkx as nx


class GraphViewerEditor:
    def __init__(self, root: tk.Tk, graph_path: str | None = None) -> None:
        self.root = root
        self.root.title("GraphML Viewer / Editor")
        self.graph: nx.Graph | None = None
        self.graph_path: Path | None = None
        self.positions: dict[str, tuple[float, float]] = {}
        self.selected_node: str | None = None

        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.world_min_x = 0.0
        self.world_max_x = 1.0
        self.world_min_y = 0.0
        self.world_max_y = 1.0
        self.pan_anchor: tuple[int, int] | None = None

        self.path_var = tk.StringVar(value=graph_path or "")
        self.status_var = tk.StringVar(value="Load a GraphML file to begin.")
        self.selected_var = tk.StringVar(value="Selected node: none")
        self.search_var = tk.StringVar()
        self.add_id_var = tk.StringVar()
        self.add_x_var = tk.StringVar()
        self.add_y_var = tk.StringVar()

        self._build_ui()
        if graph_path:
            self.load_graph(graph_path)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=8)
        container.pack(fill="both", expand=True)

        top = ttk.Frame(container)
        top.pack(fill="x")

        ttk.Label(top, text="GraphML").pack(side="left")
        ttk.Entry(top, textvariable=self.path_var, width=80).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(top, text="Browse", command=self.browse_graph).pack(side="left")
        ttk.Button(top, text="Load", command=self.load_from_entry).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="Save As", command=self.save_graph_as).pack(side="left", padx=(6, 0))

        body = ttk.Frame(container)
        body.pack(fill="both", expand=True, pady=(8, 0))

        left = ttk.Frame(body)
        left.pack(side="left", fill="y")

        ttk.Label(left, textvariable=self.selected_var).pack(anchor="w")
        ttk.Label(left, textvariable=self.status_var, wraplength=300).pack(anchor="w", pady=(4, 8))

        search_frame = ttk.LabelFrame(left, text="Find Node", padding=8)
        search_frame.pack(fill="x", pady=(0, 8))
        ttk.Entry(search_frame, textvariable=self.search_var).pack(fill="x")
        ttk.Button(search_frame, text="Find / Select", command=self.find_node).pack(fill="x", pady=(6, 0))
        ttk.Button(search_frame, text="Remove Selected Node", command=self.remove_selected_node).pack(fill="x", pady=(6, 0))

        add_frame = ttk.LabelFrame(left, text="Add Node", padding=8)
        add_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(add_frame, text="Node id").pack(anchor="w")
        ttk.Entry(add_frame, textvariable=self.add_id_var).pack(fill="x")
        ttk.Label(add_frame, text="x").pack(anchor="w", pady=(6, 0))
        ttk.Entry(add_frame, textvariable=self.add_x_var).pack(fill="x")
        ttk.Label(add_frame, text="y").pack(anchor="w", pady=(6, 0))
        ttk.Entry(add_frame, textvariable=self.add_y_var).pack(fill="x")
        ttk.Button(add_frame, text="Add Node", command=self.add_node).pack(fill="x", pady=(6, 0))

        view_frame = ttk.LabelFrame(left, text="View", padding=8)
        view_frame.pack(fill="x")
        ttk.Button(view_frame, text="Fit", command=self.fit_view).pack(fill="x")
        ttk.Button(view_frame, text="Zoom In", command=lambda: self.zoom(1.25)).pack(fill="x", pady=(6, 0))
        ttk.Button(view_frame, text="Zoom Out", command=lambda: self.zoom(0.8)).pack(fill="x", pady=(6, 0))

        self.canvas = tk.Canvas(body, bg="white", width=900, height=700, highlightthickness=1)
        self.canvas.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_pan_move)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def browse_graph(self) -> None:
        path = filedialog.askopenfilename(
            title="Open GraphML",
            filetypes=[("GraphML files", "*.graphml"), ("All files", "*.*")],
        )
        if path:
            self.path_var.set(path)

    def load_from_entry(self) -> None:
        path = self.path_var.get().strip()
        if not path:
            messagebox.showerror("Load GraphML", "Please provide a GraphML path.")
            return
        self.load_graph(path)

    def load_graph(self, path_str: str) -> None:
        path = Path(path_str)
        if not path.exists():
            messagebox.showerror("Load GraphML", f"File not found:\n{path}")
            return
        try:
            graph = nx.read_graphml(path)
        except Exception as exc:
            messagebox.showerror("Load GraphML", f"Failed to read GraphML:\n{exc}")
            return

        self.graph = graph
        self.graph_path = path
        self.positions = self._build_positions(graph)
        self.selected_node = None
        self._update_world_bounds()
        self.fit_view()
        self._set_status(
            f"Loaded {path.name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges."
        )

    def save_graph_as(self) -> None:
        if self.graph is None:
            messagebox.showerror("Save GraphML", "No graph loaded.")
            return
        default_name = self.graph_path.name if self.graph_path else "edited.graphml"
        path = filedialog.asksaveasfilename(
            title="Save GraphML As",
            defaultextension=".graphml",
            initialfile=default_name,
            filetypes=[("GraphML files", "*.graphml"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            nx.write_graphml(self.graph, path)
        except Exception as exc:
            messagebox.showerror("Save GraphML", f"Failed to write GraphML:\n{exc}")
            return
        self._set_status(f"Saved graph to {path}")

    def find_node(self) -> None:
        if self.graph is None:
            return
        node_id = self.search_var.get().strip()
        if node_id not in self.graph:
            self._set_status(f"Node {node_id!r} not found.")
            return
        self.selected_node = node_id
        self.selected_var.set(f"Selected node: {node_id}")
        self.center_on_node(node_id)
        self.redraw()

    def remove_selected_node(self) -> None:
        if self.graph is None or self.selected_node is None:
            return
        node_id = self.selected_node
        if not messagebox.askyesno("Remove Node", f"Remove node {node_id}?"):
            return
        self.graph.remove_node(node_id)
        self.positions.pop(node_id, None)
        self.selected_node = None
        self._update_world_bounds()
        self.fit_view()
        self._set_status(f"Removed node {node_id}.")

    def add_node(self) -> None:
        if self.graph is None:
            return
        node_id = self.add_id_var.get().strip()
        if not node_id:
            messagebox.showerror("Add Node", "Node id is required.")
            return
        if node_id in self.graph:
            messagebox.showerror("Add Node", f"Node {node_id!r} already exists.")
            return
        try:
            x = float(self.add_x_var.get().strip())
            y = float(self.add_y_var.get().strip())
        except ValueError:
            messagebox.showerror("Add Node", "x and y must be valid numbers.")
            return

        self.graph.add_node(node_id, x=x, y=y, street_count=0)
        self.positions[node_id] = (x, y)
        self.selected_node = node_id
        self._update_world_bounds()
        self.center_on_node(node_id)
        self.redraw()
        self.selected_var.set(f"Selected node: {node_id}")
        self._set_status(f"Added node {node_id} at ({x}, {y}).")

    def on_canvas_resize(self, _event: tk.Event) -> None:
        if self.graph is not None:
            self.redraw()

    def on_left_click(self, event: tk.Event) -> None:
        if self.graph is None:
            return
        node_id = self.find_nearest_node(event.x, event.y)
        if node_id is None:
            return
        self.selected_node = node_id
        attrs = self.graph.nodes[node_id]
        self.selected_var.set(f"Selected node: {node_id}")
        self._set_status(
            f"Node {node_id} | x={attrs.get('x')} | y={attrs.get('y')} | degree={self.graph.degree[node_id]}"
        )
        self.redraw()

    def on_pan_start(self, event: tk.Event) -> None:
        self.pan_anchor = (event.x, event.y)

    def on_pan_move(self, event: tk.Event) -> None:
        if self.pan_anchor is None:
            return
        dx = event.x - self.pan_anchor[0]
        dy = event.y - self.pan_anchor[1]
        self.offset_x += dx
        self.offset_y += dy
        self.pan_anchor = (event.x, event.y)
        self.redraw()

    def on_mousewheel(self, event: tk.Event) -> None:
        factor = 1.1 if event.delta > 0 else 0.9
        self.zoom(factor, anchor=(event.x, event.y))

    def zoom(self, factor: float, anchor: tuple[int, int] | None = None) -> None:
        if self.graph is None:
            return
        if anchor is None:
            anchor = (self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2)

        before_x, before_y = self.screen_to_world(*anchor)
        self.scale *= factor
        after_x, after_y = self.screen_to_world(*anchor)
        self.offset_x += (after_x - before_x) * self.scale
        self.offset_y -= (after_y - before_y) * self.scale
        self.redraw()

    def fit_view(self) -> None:
        if self.graph is None:
            return
        width = max(self.canvas.winfo_width(), 400)
        height = max(self.canvas.winfo_height(), 400)
        world_w = max(self.world_max_x - self.world_min_x, 1e-9)
        world_h = max(self.world_max_y - self.world_min_y, 1e-9)
        margin = 30
        self.scale = min((width - 2 * margin) / world_w, (height - 2 * margin) / world_h)
        self.offset_x = margin - self.world_min_x * self.scale
        self.offset_y = margin + self.world_max_y * self.scale
        self.redraw()

    def center_on_node(self, node_id: str) -> None:
        x, y = self.positions[node_id]
        width = max(self.canvas.winfo_width(), 400)
        height = max(self.canvas.winfo_height(), 400)
        self.offset_x = width / 2.0 - x * self.scale
        self.offset_y = height / 2.0 + y * self.scale

    def redraw(self) -> None:
        self.canvas.delete("all")
        if self.graph is None:
            return

        for u, v in self.graph.edges():
            x1, y1 = self.world_to_screen(*self.positions[u])
            x2, y2 = self.world_to_screen(*self.positions[v])
            self.canvas.create_line(x1, y1, x2, y2, fill="#a8b0ba", width=1)

        for node_id, (x, y) in self.positions.items():
            sx, sy = self.world_to_screen(x, y)
            radius = 4 if node_id == self.selected_node else 2
            color = "#d62828" if node_id == self.selected_node else "#1d4ed8"
            self.canvas.create_oval(sx - radius, sy - radius, sx + radius, sy + radius, fill=color, outline="")

        if self.selected_node is not None:
            x, y = self.positions[self.selected_node]
            sx, sy = self.world_to_screen(x, y)
            self.canvas.create_text(
                sx + 10,
                sy - 10,
                text=self.selected_node,
                fill="#111827",
                anchor="w",
                font=("Segoe UI", 10, "bold"),
            )

    def find_nearest_node(self, sx: int, sy: int) -> str | None:
        if not self.positions:
            return None
        best_node = None
        best_dist2 = float("inf")
        for node_id, (x, y) in self.positions.items():
            nx_s, ny_s = self.world_to_screen(x, y)
            dist2 = (nx_s - sx) ** 2 + (ny_s - sy) ** 2
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_node = node_id
        if best_dist2 > 16 ** 2:
            return None
        return best_node

    def world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.offset_x, self.offset_y - y * self.scale

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return (sx - self.offset_x) / self.scale, (self.offset_y - sy) / self.scale

    def _build_positions(self, graph: nx.Graph) -> dict[str, tuple[float, float]]:
        positions: dict[str, tuple[float, float]] = {}
        fallback_nodes: list[str] = []
        for node_id, attrs in graph.nodes(data=True):
            try:
                positions[node_id] = (float(attrs["x"]), float(attrs["y"]))
            except (KeyError, TypeError, ValueError):
                fallback_nodes.append(node_id)

        if fallback_nodes:
            layout = nx.spring_layout(graph, seed=42)
            for node_id in fallback_nodes:
                x, y = layout[node_id]
                graph.nodes[node_id]["x"] = float(x)
                graph.nodes[node_id]["y"] = float(y)
                positions[node_id] = (float(x), float(y))
        return positions

    def _update_world_bounds(self) -> None:
        xs = [pos[0] for pos in self.positions.values()]
        ys = [pos[1] for pos in self.positions.values()]
        self.world_min_x = min(xs)
        self.world_max_x = max(xs)
        self.world_min_y = min(ys)
        self.world_max_y = max(ys)

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive GraphML viewer/editor.")
    parser.add_argument("--graph", help="Optional GraphML file to load immediately.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = tk.Tk()
    app = GraphViewerEditor(root, graph_path=args.graph)
    root.mainloop()


if __name__ == "__main__":
    main()
