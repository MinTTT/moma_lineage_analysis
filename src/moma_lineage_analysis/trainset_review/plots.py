"""Interactive trajectory plots matching the main lineage-app visual style."""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from moma_lineage_analysis.trajectory_view_style import (
    AI_DIVISION_MARKER, AI_EXCLUSION_MARKER, APPLIED_EXCLUSION_MARKER,
    TRAJECTORY_TRACE_COLORS, trajectory_plot_runs,
)

from .evaluation import PROBABILITY_COLUMNS, threshold_pair, validate_predictions
from .review import observation_id, validate_frame

DIVISION_LABEL_MARKER = dict(symbol="triangle-up", size=10, color="#C62828",
                             line=dict(width=1.5, color="#7f1d1d"))
ORIGINAL_DIVISION_MARKER = dict(symbol="diamond-open", size=11, color="#7B3294",
                                line=dict(width=1.5))
ORIGINAL_EXCLUSION_MARKER = dict(symbol="square-open", size=11, color="#999999",
                                 line=dict(width=1.5))
SPINE_SUPLOT_TITLE = "Spine Length (Log) - Select points to exclude or restore"
AI_SUPLOT_TITLE = "AI probabilities (0–1)"


def selected_observation_ids(event: Any, allowed: Iterable[int]) -> list[int]:
    """Read Streamlit/Plotly selection customdata without converting via float.

    Accepts {selection: {points: [...]}} or {points: [...]}, with each point's
    customdata a decimal string or a one-element sequence containing that ID.
    Missing customdata is ignored (decorative traces); malformed/unknown IDs
    raise ValueError. Duplicate selections are removed while preserving order.
    """
    allowed_ids = {observation_id(value) for value in allowed}
    if event is None:
        return []
    if not isinstance(event, Mapping):
        raise ValueError("Selection event must be a mapping")
    selection = event.get("selection", event)
    if not isinstance(selection, Mapping):
        raise ValueError("Selection payload must be a mapping")
    points = selection.get("points", [])
    if not isinstance(points, (list, tuple)):
        raise ValueError("Selection points must be a sequence")
    selected = []
    for point in points:
        if not isinstance(point, Mapping):
            raise ValueError("Each selected point must be a mapping")
        if "customdata" not in point or point["customdata"] is None:
            continue
        value = point["customdata"]
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != 1:
                raise ValueError("Selected customdata must contain exactly one observation ID")
            value = value[0]
        cell_id = observation_id(value, from_browser=True)
        if cell_id not in allowed_ids:
            raise ValueError(f"Selected observation ID is outside this trajectory: {cell_id}")
        if cell_id not in selected:
            selected.append(cell_id)
    return selected


def _align_overlay(frame: pd.DataFrame, overlay: pd.DataFrame, *, probabilities: bool) -> pd.DataFrame:
    source = validate_predictions(overlay) if probabilities else validate_frame(overlay)
    key = frame.trajectory_key.iloc[0]
    source = source[source.trajectory_key == key]
    ids = [observation_id(value) for value in frame.fov_cell_index]
    overlay_ids = [observation_id(value) for value in source.fov_cell_index]
    if set(overlay_ids) != set(ids) or len(source) != len(frame):
        raise ValueError("Plot overlay must contain exactly the displayed trajectory's observation IDs")
    order = {cell_id: position for position, cell_id in enumerate(overlay_ids)}
    result = source.iloc[[order[cell_id] for cell_id in ids]].copy()
    for column in ("time_h", "spine_length"):
        if not np.array_equal(pd.to_numeric(frame[column]).to_numpy(), pd.to_numeric(result[column]).to_numpy()):
            raise ValueError(f"Plot overlay {column} does not align with displayed measurements")
    return result


def _plot_epoch(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError("plot_epoch must be a nonnegative integer")
    return int(value)


def _ids_customdata(frame: pd.DataFrame) -> list[list[str]]:
    # JS Number cannot represent all int64 IDs; decimal strings are lossless.
    return [[str(observation_id(value))] for value in frame.fov_cell_index]


def _marker(marker: Mapping[str, Any]) -> dict[str, Any]:
    return dict(marker)


def _add_markers(fig: go.Figure, frame: pd.DataFrame, name: str, marker: Mapping[str, Any],
                 *, row: int = 1, customdata: list[list[str]] | None = None) -> None:
    if frame.empty:
        return
    data = customdata if customdata is not None else _ids_customdata(frame)
    fig.add_trace(go.Scatter(
        x=pd.to_numeric(frame.time_h), y=pd.to_numeric(frame.spine_length), mode="markers",
        name=name, marker=_marker(marker), customdata=data,
        hovertemplate=("Time: %{x:.2f}h<br>Length: %{y:.2f}um"
                       "<br>ID: %{customdata[0]}<extra></extra>"),
    ), row=row, col=1)


def build_trajectory_plot(frame: pd.DataFrame, predictions: pd.DataFrame | None = None,
                          thresholds: Iterable[float] | None = None,
                          original_labels: pd.DataFrame | None = None,
                          plot_epoch: int = 0,
                          figure: go.Figure | None = None) -> go.Figure:
    """Build the shared main-app trajectory figure for one reviewed trajectory.

    Frame rows are sorted by time only for drawing. Colored spine runs use the
    shared splitter, so a line never bridges an excluded frame or a division
    boundary. Applied exclusions and division labels remain individually
    selectable at measured positions. With predictions, a second 0–1 probability
    panel is added; thresholded AI markers and dotted threshold labels appear
    only when thresholds are also supplied. original_labels is an optional
    complete pre-edit trajectory overlay, aligned strictly by integer ID.
    plot_epoch advances selectionrevision to clear stale lasso selections.
    All inputs are validated and left unchanged.
    """
    source = validate_frame(frame)
    if source.trajectory_key.nunique() != 1:
        raise ValueError("Plot one trajectory at a time")
    revision = _plot_epoch(plot_epoch)
    selected = None if thresholds is None else threshold_pair(thresholds)
    ordered = source.sort_values("time_h", kind="stable").reset_index(drop=True)
    # Both overlays are returned in the displayed frame's order; only ID/measurement
    # equality is accepted, so a stale or partial prediction can never be drawn.
    overlay = None if predictions is None else _align_overlay(
        ordered, predictions, probabilities=True).reset_index(drop=True)
    original = None if original_labels is None else _align_overlay(
        ordered, original_labels, probabilities=False).reset_index(drop=True)


    excluded_ids = [observation_id(v) for v, flag in zip(ordered.fov_cell_index, ordered.excluded_point) if flag]
    division_ids = [observation_id(v) for v, flag in zip(ordered.fov_cell_index, ordered.division_point) if flag]
    runs = trajectory_plot_runs(ordered, excluded_ids, division_ids, presorted=True)
    has_ai = overlay is not None
    rows = 2 if has_ai else 1
    existing_meta = getattr(figure.layout, 'meta', None) if figure is not None else None
    reuse = isinstance(existing_meta, dict) and existing_meta.get('moma_rows') == rows
    if reuse:
        fig = figure
        # Keep the existing layout/axes; replace only the plotted data. Threshold
        # lines and their annotations are regenerated below, so remove them first.
        base_annotations = int(existing_meta.get('moma_base_annotations', 0))
        fig.data = ()
        fig.layout.shapes = ()
        fig.layout.annotations = tuple(fig.layout.annotations[:base_annotations])
    else:
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=(SPINE_SUPLOT_TITLE, AI_SUPLOT_TITLE) if has_ai
                            else (SPINE_SUPLOT_TITLE,))
        base_annotations = len(fig.layout.annotations)
    for group, run in runs:
        color = TRAJECTORY_TRACE_COLORS[group % len(TRAJECTORY_TRACE_COLORS)]
        fig.add_trace(go.Scatter(
            x=pd.to_numeric(run.time_h), y=pd.to_numeric(run.spine_length), mode="markers+lines",
            name=f"Cycle {group + 1}", showlegend=False,
            line=dict(color=color), marker=dict(color=color), customdata=_ids_customdata(run),
            hovertemplate=("Time: %{x:.2f}h<br>Length: %{y:.2f}um"
                           "<br>ID: %{customdata[0]}<extra></extra>"),
        ), row=1, col=1)

    excluded = ordered.loc[ordered.excluded_point.to_numpy(bool)].copy()
    _add_markers(fig, excluded, "Applied exclusion", APPLIED_EXCLUSION_MARKER)
    divisions = ordered.loc[ordered.division_point.to_numpy(bool)].copy()
    _add_markers(fig, divisions, "Division label", DIVISION_LABEL_MARKER)
    if original is not None:
        _add_markers(fig, original.loc[original.division_point.to_numpy(bool)].copy(),
                     "Original division", ORIGINAL_DIVISION_MARKER)
        _add_markers(fig, original.loc[original.excluded_point.to_numpy(bool)].copy(),
                     "Original exclusion", ORIGINAL_EXCLUSION_MARKER)
    if has_ai:
        probability_runs = trajectory_plot_runs(ordered, excluded_ids, division_ids, presorted=True)
        for group, run in probability_runs:
            probability = overlay.loc[list(run.index), list(PROBABILITY_COLUMNS)].to_numpy(float)
            color = TRAJECTORY_TRACE_COLORS[group % len(TRAJECTORY_TRACE_COLORS)]
            for index, name, dash in ((0, "Division probability", "solid"),
                                      (1, "Exclusion probability", "dot")):
                fig.add_trace(go.Scatter(
                    x=pd.to_numeric(run.time_h), y=probability[:, index], mode="markers+lines",
                    name=name, showlegend=group == 0,
                    line=dict(color=color, dash=dash), marker=dict(color=color, size=4),
                    customdata=_ids_customdata(run),
                    hovertemplate=("Time: %{x:.2f}h<br>Probability: %{y:.3f}"
                                   "<br>ID: %{customdata[0]}<extra></extra>"),
                ), row=2, col=1)
        if selected is not None:
            probs = overlay.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(float)
            division = ordered.loc[probs[:, 0] >= selected[0]].copy()
            exclusion = ordered.loc[probs[:, 1] >= selected[1]].copy()
            _add_markers(fig, division, "AI division", AI_DIVISION_MARKER)
            _add_markers(fig, exclusion, "AI exclusion", AI_EXCLUSION_MARKER)
            for index, label, color, dash in ((0, "Division", "#D55E00", "dot"),
                                              (1, "Exclusion", "#333333", "dash")):
                fig.add_hline(y=float(selected[index]), line_dash=dash, line_color=color,
                              annotation_text=f"{label} threshold {selected[index]:.2f}",
                              annotation_position="top left", row=2, col=1)
        fig.update_yaxes(title_text="Probability", range=[0, 1], row=2, col=1)
    fig.update_yaxes(type="log", title_text="Spine length (µm)", row=1, col=1)
    fig.update_xaxes(title_text="Time (h)", row=rows, col=1)
    fig.update_layout(
        template="plotly_white",
        title=dict(text=str(ordered.trajectory_key.iloc[0]), y=0.99, x=0.05),
        height=300 + 300 * rows,
        hovermode="x unified",
        dragmode="lasso",
        selectionrevision=revision,
        uirevision=str(ordered.trajectory_key.iloc[0]),
        margin=dict(l=70, r=25, t=75, b=90),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0,
                    entrywidth=165, entrywidthmode="pixels"),
        meta={'moma_rows': rows, 'moma_base_annotations': base_annotations},
    )
    return fig
