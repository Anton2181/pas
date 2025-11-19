#!/usr/bin/env python3
"""Produce a set of graphs that highlight component exclusivity relations."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

import encode_sat_from_components as encoder

LAYOUT_CHOICES = ("grid", "spring", "component")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize component conflicts")
    ap.add_argument("--components", default="components_all.csv", type=Path)
    ap.add_argument("--backend", default="backend.csv", type=Path)
    ap.add_argument(
        "--out-prefix",
        "--out",
        dest="out_prefix",
        default=Path("components_graph"),
        type=Path,
        help="Prefix for generated graph files (suffixes are added per layout)",
    )
    ap.add_argument(
        "--layouts",
        nargs="+",
        default=list(LAYOUT_CHOICES),
        choices=LAYOUT_CHOICES,
        help="One or more layout names to render",
    )
    ap.add_argument("--min-weight", type=float, default=0.5, help="Minimum edge alpha when rendering")
    ap.add_argument("--dpi", type=int, default=200, help="Output DPI")
    return ap.parse_args()


def build_conflicts(
    comps: Iterable[encoder.CompRow],
    exclusions: Dict[str, set],
) -> List[Tuple[str, str, Dict[str, str]]]:
    by_week_day: Dict[Tuple[int, str], List[encoder.CompRow]] = defaultdict(list)
    for row in comps:
        by_week_day[(row.week_num, row.day)].append(row)

    edges: List[Tuple[str, str, Dict[str, str]]] = []
    for (week, day), rows in by_week_day.items():
        for i in range(len(rows)):
            a = rows[i]
            names_a = set(a.names)
            for j in range(i + 1, len(rows)):
                b = rows[j]
                names_b = set(b.names)
                if _conflicts(names_a, names_b, exclusions):
                    edges.append((a.cid, b.cid, {"week": a.week_label, "day": day}))
    return edges


def _conflicts(names_a: set[str], names_b: set[str], exclusions: Dict[str, set]) -> bool:
    for task_a in names_a:
        forbidden = exclusions.get(task_a, set())
        if forbidden & names_b:
            return True
    for task_b in names_b:
        forbidden = exclusions.get(task_b, set())
        if forbidden & names_a:
            return True
    return False


def _build_graph(
    comps: List[encoder.CompRow],
    edges: List[Tuple[str, str, Dict[str, str]]],
) -> nx.Graph:
    graph = nx.Graph()
    for comp in comps:
        label = f"{comp.cid}\n{comp.week_label} {comp.day}"
        graph.add_node(
            comp.cid,
            label=label,
            priority=comp.priority,
            day=comp.day,
            week=comp.week_num,
        )

    for src, dst, meta in edges:
        if src in graph and dst in graph:
            graph.add_edge(src, dst, **meta)

    if not graph.nodes:
        raise RuntimeError("No components to visualize")
    return graph


def _layout_grid(graph: nx.Graph) -> Dict[str, Tuple[float, float]]:
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    day_index = {day: idx for idx, day in enumerate(day_order)}
    fallback = len(day_order)
    positions: Dict[str, Tuple[float, float]] = {}
    per_slot: Dict[Tuple[int, str], int] = defaultdict(int)
    for cid in graph.nodes:
        week = graph.nodes[cid]["week"]
        day = graph.nodes[cid]["day"]
        slot = (week, day)
        per_slot[slot] += 1
        offset = (per_slot[slot] - 1) * 0.15
        x = week + offset
        y = day_index.get(day, fallback)
        positions[cid] = (x, y)
    return positions


def _layout_spring(graph: nx.Graph) -> Dict[str, Tuple[float, float]]:
    if len(graph.nodes) == 1:
        return {next(iter(graph.nodes)): (0.0, 0.0)}
    return nx.spring_layout(graph, seed=42)


def _layout_components(graph: nx.Graph) -> Dict[str, Tuple[float, float]]:
    positions: Dict[str, Tuple[float, float]] = {}
    comps = list(nx.connected_components(graph))
    if not comps:
        comps = [[n] for n in graph.nodes]
    spacing = 4.0
    x_cursor = 0.0
    for comp_nodes in sorted(comps, key=len, reverse=True):
        sorted_nodes = sorted(comp_nodes)
        for idx, node in enumerate(sorted_nodes):
            positions[node] = (x_cursor, -idx)
        x_cursor += spacing
    return positions


LAYOUT_FNS = {
    "grid": _layout_grid,
    "spring": _layout_spring,
    "component": _layout_components,
}


def _day_palette(graph: nx.Graph) -> Dict[str, str]:
    days = sorted({graph.nodes[n]["day"] for n in graph.nodes})
    cmap = plt.cm.get_cmap("tab10", max(3, len(days)))
    return {day: matplotlib.colors.rgb2hex(cmap(idx)) for idx, day in enumerate(days)}


def draw_graph_variants(
    comps: List[encoder.CompRow],
    edges: List[Tuple[str, str, Dict[str, str]]],
    out_prefix: Path,
    *,
    layouts: List[str],
    min_weight: float,
    dpi: int,
) -> List[Path]:
    graph = _build_graph(comps, edges)
    palette = _day_palette(graph)
    generated: List[Path] = []

    prefix = out_prefix
    if prefix.suffix:
        prefix = prefix.with_suffix("")

    for layout in layouts:
        layout_fn = LAYOUT_FNS[layout]
        positions = layout_fn(graph)
        out_path = prefix.parent / f"{prefix.name}_{layout}.png"
        _render_graph(graph, positions, out_path, palette, min_weight=min_weight, dpi=dpi, layout_name=layout)
        generated.append(out_path)
    return generated


def _render_graph(
    graph: nx.Graph,
    positions: Dict[str, Tuple[float, float]],
    out_path: Path,
    palette: Dict[str, str],
    *,
    min_weight: float,
    dpi: int,
    layout_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 9))
    node_colors = [palette.get(graph.nodes[n]["day"], "#666666") for n in graph.nodes]
    node_sizes = [1000 if graph.nodes[n]["priority"] else 700 for n in graph.nodes]
    nx.draw_networkx_nodes(graph, positions, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax, linewidths=0.5, edgecolors="#2f2f2f")
    labels = {n: graph.nodes[n]["label"] for n in graph.nodes}
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=8, ax=ax, bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", alpha=0.7))

    if graph.edges:
        alphas = max(min_weight, 0.3)
        widths = [1.5 for _ in graph.edges]
        nx.draw_networkx_edges(graph, positions, width=widths, alpha=alphas, ax=ax, edge_color="#555555")

    ax.set_title(f"Component conflicts ({layout_name} layout)")
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    comps, _, _ = encoder.load_components(args.components)
    backend_matrix = encoder.read_csv_matrix(args.backend)
    _, _, exclusions, _, _, _, _ = encoder.load_backend_roles_and_maps(backend_matrix)
    edges = build_conflicts(comps, exclusions)
    outputs = draw_graph_variants(
        comps,
        edges,
        args.out_prefix,
        layouts=args.layouts,
        min_weight=args.min_weight,
        dpi=args.dpi,
    )
    for path in outputs:
        print(f"Wrote graph to {path}")


if __name__ == "__main__":
    main()
