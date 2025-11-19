#!/usr/bin/env python3
"""Produce a set of graphs that highlight component exclusivity relations."""
from __future__ import annotations

import argparse
import math
import statistics
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import axes as mpl_axes
from matplotlib import colors as mpl_colors
from matplotlib.lines import Line2D
import networkx as nx

import encode_sat_from_components as encoder


@dataclass
class CandidateColoring:
    cmap: matplotlib.colors.Colormap
    norm: mpl_colors.Normalize

LAYOUT_CHOICES = ("grid", "spring", "component", "calendar")
LABEL_CHOICES = ("full", "short", "none")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize component conflicts")
    ap.add_argument("--components", default="components_all.csv", type=Path)
    ap.add_argument("--backend", default="backend.csv", type=Path)
    ap.add_argument(
        "--out-dir",
        "--graphs-dir",
        dest="out_dir",
        default=Path("components_graphs"),
        type=Path,
        help="Directory for generated graph files",
    )
    ap.add_argument(
        "--out-prefix",
        "--out",
        dest="out_prefix",
        default="components_graph",
        type=str,
        help="Base filename prefix for graph images (suffixes are added per layout)",
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
    ap.add_argument(
        "--analysis-dir",
        "--analysis-prefix",
        dest="analysis_dir",
        type=Path,
        default=Path("components_analysis"),
        help="Directory for non-graph analysis charts",
    )
    ap.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip generating histogram/heatmap/scatter plots",
    )
    ap.add_argument(
        "--label-mode",
        choices=LABEL_CHOICES,
        default="short",
        help="Label verbosity (full calendar label, CID only, or none)",
    )
    ap.add_argument(
        "--max-label-chars",
        type=int,
        default=32,
        help="Clamp labels to this many characters when label-mode is short",
    )
    ap.add_argument(
        "--candidate-source",
        choices=("role", "all"),
        default="role",
        help="Which candidate list to use when computing availability counts",
    )
    return ap.parse_args()


def _candidate_count(row: encoder.CompRow, source: str) -> int:
    if source == "all":
        pool = row.candidates_all or row.candidates_role
    else:
        pool = row.candidates_role or row.candidates_all
    return len(pool)


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
    candidate_counts: Dict[str, int],
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
            cand_count=candidate_counts.get(comp.cid, 0),
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


def _candidate_coloring(graph: nx.Graph) -> CandidateColoring:
    counts = [graph.nodes[n].get("cand_count", 0) for n in graph.nodes]
    if not counts:
        raise RuntimeError("Graph has no nodes")
    vmin = min(counts)
    vmax = max(counts)
    if vmin == vmax:
        vmax = vmin + 1
    cmap = plt.get_cmap("plasma")
    norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    return CandidateColoring(cmap=cmap, norm=norm)


def _day_palette(graph: nx.Graph) -> Dict[str, str]:
    days = sorted({graph.nodes[n]["day"] for n in graph.nodes})
    cmap = plt.get_cmap("tab10", max(3, len(days)))
    return {day: matplotlib.colors.rgb2hex(cmap(idx)) for idx, day in enumerate(days)}


def _format_labels(graph: nx.Graph, mode: str, max_chars: int) -> Dict[str, str]:
    if mode == "none":
        return {}
    labels: Dict[str, str] = {}
    for node in graph.nodes:
        text = graph.nodes[node]["label"]
        if mode == "short":
            text = text.split("\n", 1)[0]
            if len(text) > max_chars:
                text = text[: max_chars - 1] + "…"
        labels[node] = text
    return labels


def draw_graph_variants(
    graph: nx.Graph,
    out_dir: Path,
    out_prefix: str,
    *,
    layouts: List[str],
    min_weight: float,
    dpi: int,
    label_mode: str,
    max_label_chars: int,
) -> List[Path]:
    palette = _day_palette(graph)
    color_config = _candidate_coloring(graph)
    generated: List[Path] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(out_prefix).stem or "components_graph"

    for layout in layouts:
        out_path = out_dir / f"{prefix}_{layout}.png"
        if layout == "calendar":
            _render_calendar(
                graph,
                out_path,
                palette,
                color_config,
                dpi=dpi,
                label_mode=label_mode,
                max_label_chars=max_label_chars,
                min_weight=min_weight,
            )
        else:
            layout_fn = LAYOUT_FNS[layout]
            positions = layout_fn(graph)
            _render_graph(
                graph,
                positions,
                out_path,
                palette,
                color_config,
                min_weight=min_weight,
                dpi=dpi,
                layout_name=layout,
                label_mode=label_mode,
                max_label_chars=max_label_chars,
            )
        generated.append(out_path)
    return generated


def _render_graph(
    graph: nx.Graph,
    positions: Dict[str, Tuple[float, float]],
    out_path: Path,
    palette: Dict[str, str],
    color_config: CandidateColoring,
    *,
    min_weight: float,
    dpi: int,
    layout_name: str,
    label_mode: str,
    max_label_chars: int,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 9))
    _draw_graph_on_axis(
        ax,
        graph,
        positions,
        palette,
        color_config,
        min_weight=min_weight,
        label_mode=label_mode,
        max_label_chars=max_label_chars,
        show_colorbar=True,
    )
    ax.set_title(f"Component conflicts ({layout_name} layout)")
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _draw_graph_on_axis(
    ax: plt.Axes,
    graph: nx.Graph,
    positions: Dict[str, Tuple[float, float]],
    palette: Dict[str, str],
    color_config: CandidateColoring,
    *,
    min_weight: float,
    label_mode: str,
    max_label_chars: int,
    show_colorbar: bool,
) -> None:
    fig = ax.get_figure()
    node_sizes = [1100 if graph.nodes[n]["priority"] else 750 for n in graph.nodes]
    node_colors = [color_config.cmap(color_config.norm(graph.nodes[n]["cand_count"])) for n in graph.nodes]
    border_colors = [palette.get(graph.nodes[n]["day"], "#2f2f2f") for n in graph.nodes]
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.92,
        ax=ax,
        linewidths=1.2,
        edgecolors=border_colors,
    )

    labels = _format_labels(graph, mode=label_mode, max_chars=max_label_chars)
    if labels:
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=labels,
            font_size=8,
            font_weight="bold" if label_mode == "short" else "normal",
            ax=ax,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", alpha=0.65, linewidth=0),
        )

    if graph.edges:
        alpha = max(min_weight, 0.25)
        widths = [1.2 for _ in graph.edges]
        nx.draw_networkx_edges(graph, positions, width=widths, alpha=alpha, ax=ax, edge_color="#555555")

    handles = _day_handles(palette)
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, title="Day (border color)")

    if show_colorbar:
        sm = plt.cm.ScalarMappable(norm=color_config.norm, cmap=color_config.cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Eligible candidates")


def _day_handles(palette: Dict[str, str]) -> List[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#ffffff",
            markeredgecolor=color,
            markeredgewidth=1.5,
            label=day,
        )
        for day, color in palette.items()
    ]


def _render_calendar(
    graph: nx.Graph,
    out_path: Path,
    palette: Dict[str, str],
    color_config: CandidateColoring,
    *,
    dpi: int,
    label_mode: str,
    max_label_chars: int,
    min_weight: float = 0.35,
) -> None:
    days = sorted({graph.nodes[n]["day"] for n in graph.nodes})
    fig_height = max(3.5, 2.2 * len(days))
    fig, axes = plt.subplots(len(days), 1, figsize=(14, fig_height), sharex=True)
    if isinstance(axes, mpl_axes.Axes):
        axes_list = [axes]
    elif isinstance(axes, (list, tuple)):
        axes_list = list(axes)
    else:
        axes_list = list(axes.ravel())

    for day, ax in zip(days, axes_list):
        nodes = [n for n in graph.nodes if graph.nodes[n]["day"] == day]
        sub = graph.subgraph(nodes).copy()
        if not nodes:
            ax.set_axis_off()
            continue
        positions = _layout_calendar(sub)
        _draw_graph_on_axis(
            ax,
            sub,
            positions,
            palette,
            color_config,
            min_weight=min_weight,
            label_mode=label_mode,
            max_label_chars=max_label_chars,
            show_colorbar=False,
        )
        ax.set_title(day, loc="left", fontsize=11)
        ax.set_ylabel("slot")
        ax.set_axis_on()
        ax.grid(True, axis="x", linestyle="--", alpha=0.2)

    fig.supxlabel("Week")
    fig.supylabel("Slot per week")
    sm = plt.cm.ScalarMappable(norm=color_config.norm, cmap=color_config.cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes_list, fraction=0.018, pad=0.01, label="Eligible candidates")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _layout_calendar(graph: nx.Graph) -> Dict[str, Tuple[float, float]]:
    per_week: Dict[int, int] = defaultdict(int)
    positions: Dict[str, Tuple[float, float]] = {}
    for node in sorted(graph.nodes, key=lambda n: (graph.nodes[n]["week"], n)):
        week = graph.nodes[node]["week"]
        per_week[week] += 1
        y = per_week[week]
        positions[node] = (week, y)
    return positions


def generate_analysis_charts(
    comps: List[encoder.CompRow],
    candidate_counts: Dict[str, int],
    graph: nx.Graph,
    out_dir: Path,
    out_prefix: str,
    *,
    dpi: int,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(out_prefix).stem or "components_analysis"
    outputs: List[Path] = []
    counts = [candidate_counts.get(row.cid, 0) for row in comps]
    hist_path = out_dir / f"{prefix}_candidate_hist.png"
    _plot_candidate_hist(counts, hist_path, dpi=dpi)
    outputs.append(hist_path)

    heatmap_path = out_dir / f"{prefix}_week_day_heatmap.png"
    _plot_week_day_heatmap(comps, candidate_counts, heatmap_path, dpi=dpi)
    outputs.append(heatmap_path)

    scatter_path = out_dir / f"{prefix}_degree_scatter.png"
    _plot_degree_scatter(graph, candidate_counts, scatter_path, dpi=dpi)
    outputs.append(scatter_path)
    return outputs


def _plot_candidate_hist(counts: List[int], out_path: Path, *, dpi: int) -> None:
    if not counts:
        return
    max_count = max(counts)
    bins = range(0, max_count + 2)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(counts, bins=bins, color="#6baed6", edgecolor="#1f1f1f", alpha=0.85)
    ax.set_xlabel("Eligible candidates per component")
    ax.set_ylabel("Component count")
    ax.set_title("Candidate availability distribution")
    try:
        median = statistics.median(counts)
        quartiles = statistics.quantiles(counts, n=4)
        q1, q3 = quartiles[0], quartiles[2]
        for value, label, color in (
            (median, "median", "#cb181d"),
            (q1, "Q1", "#756bb1"),
            (q3, "Q3", "#31a354"),
        ):
            ax.axvline(value, color=color, linestyle="--", linewidth=1, label=label)
    except statistics.StatisticsError:
        pass
    ax.legend(loc="upper right", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_week_day_heatmap(
    comps: List[encoder.CompRow],
    candidate_counts: Dict[str, int],
    out_path: Path,
    *,
    dpi: int,
) -> None:
    if not comps:
        return
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weeks = sorted({row.week_num for row in comps})
    if not weeks:
        return
    matrix: List[List[float]] = []
    for day in day_order:
        row_values: List[float] = []
        for week in weeks:
            values = [candidate_counts.get(c.cid, 0) for c in comps if c.day == day and c.week_num == week]
            avg = sum(values) / len(values) if values else float("nan")
            row_values.append(avg)
        matrix.append(row_values)

    fig, ax = plt.subplots(figsize=(max(8, len(weeks) * 0.8), 4))
    cmap = plt.get_cmap("YlGnBu")
    cax = ax.imshow(matrix, aspect="auto", cmap=cmap)
    for y, day in enumerate(day_order):
        for x, week in enumerate(weeks):
            value = matrix[y][x]
            if math.isnan(value):
                continue
            ax.text(
                x,
                y,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="#000000" if value < (cax.get_clim()[1] * 0.6) else "#ffffff",
                fontsize=8,
            )
    ax.set_xticks(range(len(weeks)), [str(w) for w in weeks])
    ax.set_yticks(range(len(day_order)), day_order)
    ax.set_xlabel("Week number")
    ax.set_ylabel("Day")
    ax.set_title("Average candidate availability per day/week")
    fig.colorbar(cax, ax=ax, label="Avg candidates")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_degree_scatter(
    graph: nx.Graph,
    candidate_counts: Dict[str, int],
    out_path: Path,
    *,
    dpi: int,
) -> None:
    if not graph.nodes:
        return
    degrees = dict(graph.degree())
    xs = [candidate_counts.get(node, 0) for node in graph.nodes]
    ys = [degrees.get(node, 0) for node in graph.nodes]
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(xs, ys, c=ys, cmap="PuRd", edgecolor="#2f2f2f", alpha=0.85)
    ax.set_xlabel("Eligible candidates")
    ax.set_ylabel("Conflict degree")
    ax.set_title("Conflicts vs. candidate depth")

    top_nodes = sorted(graph.nodes, key=lambda n: degrees.get(n, 0), reverse=True)[:5]
    for node in top_nodes:
        ax.annotate(
            node,
            (candidate_counts.get(node, 0), degrees.get(node, 0)),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", alpha=0.7, linewidth=0),
        )

    fig.colorbar(scatter, ax=ax, label="Degree")
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
    candidate_counts = {row.cid: _candidate_count(row, args.candidate_source) for row in comps}
    graph = _build_graph(comps, edges, candidate_counts)
    outputs = draw_graph_variants(
        graph,
        args.out_dir,
        args.out_prefix,
        layouts=args.layouts,
        min_weight=args.min_weight,
        dpi=args.dpi,
        label_mode=args.label_mode,
        max_label_chars=args.max_label_chars,
    )
    for path in outputs:
        print(f"Wrote graph to {path}")

    if not args.skip_analysis:
        analysis_paths = generate_analysis_charts(
            comps,
            candidate_counts,
            graph,
            args.analysis_dir,
            args.out_prefix,
            dpi=args.dpi,
        )
        for path in analysis_paths:
            print(f"Wrote analysis chart to {path}")


if __name__ == "__main__":
    main()
