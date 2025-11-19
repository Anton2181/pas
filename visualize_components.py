#!/usr/bin/env python3
"""Produce a graph that highlights component exclusivity relations."""
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize component conflicts")
    ap.add_argument("--components", default="components_all.csv", type=Path)
    ap.add_argument("--backend", default="backend.csv", type=Path)
    ap.add_argument("--out", default=Path("components_graph.png"), type=Path)
    ap.add_argument("--min-weight", type=float, default=0.5, help="Minimum edge alpha when rendering")
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


def draw_graph(
    comps: List[encoder.CompRow],
    edges: List[Tuple[str, str, Dict[str, str]]],
    out_path: Path,
    *,
    min_weight: float = 0.5,
) -> None:
    G = nx.Graph()
    comp_by_id = {c.cid: c for c in comps}
    for comp in comps:
        label = f"{comp.cid}\n{comp.week_label} {comp.day}"
        G.add_node(comp.cid, label=label, priority=comp.priority)

    for src, dst, meta in edges:
        G.add_edge(src, dst, **meta)

    if not G.nodes:
        raise RuntimeError("No components to visualize")

    days = sorted({c.day for c in comps})
    day_index = {day: idx for idx, day in enumerate(days)}
    pos = {cid: (comp_by_id[cid].week_num, day_index.get(comp_by_id[cid].day, 0)) for cid in G.nodes}

    plt.figure(figsize=(12, 8))
    node_colors = ["#ff6f61" if G.nodes[n]["priority"] else "#4e79a7" for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900, alpha=0.85)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes}, font_size=8)

    if edges:
        weights = [max(min_weight, 0.5) for _ in edges]
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=weights)

    plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    comps, _, _ = encoder.load_components(args.components)
    backend_matrix = encoder.read_csv_matrix(args.backend)
    _, _, exclusions, _, _, _, _ = encoder.load_backend_roles_and_maps(backend_matrix)
    edges = build_conflicts(comps, exclusions)
    draw_graph(comps, edges, args.out, min_weight=args.min_weight)
    print(f"Wrote graph to {args.out}")


if __name__ == "__main__":
    main()
