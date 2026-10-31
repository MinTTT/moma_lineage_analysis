"""Main-app-style trajectory plot regressions and opaque selection contracts."""
import numpy as np
import pandas as pd
import pytest

from moma_lineage_analysis.trajectory_view_style import (
    AI_DIVISION_MARKER, AI_EXCLUSION_MARKER, APPLIED_EXCLUSION_MARKER,
    TRAJECTORY_TRACE_COLORS,
)
from moma_lineage_analysis.trainset_review.plots import (
    AI_SUPLOT_TITLE, DIVISION_LABEL_MARKER, ORIGINAL_DIVISION_MARKER,
    ORIGINAL_EXCLUSION_MARKER, SPINE_SUPLOT_TITLE, build_trajectory_plot,
    selected_observation_ids,
)

BIG = 2**53 + 101


def frame(n=12):
    ids = np.arange(BIG, BIG + n, dtype=np.int64)
    return pd.DataFrame({
        "trajectory_key": ["a"] * n,
        "fov_cell_index": pd.Series(ids, dtype="int64"),
        "time_h": np.arange(n) * 0.1,
        "spine_length": 2.0 + np.arange(n) * 0.4,
        "division_point": [i in {3, 8} for i in range(n)],
        "excluded_point": [i in {5, 6} for i in range(n)],
    })


def probabilities(data):
    n = len(data)
    return data.assign(
        pred_division_probability=(np.arange(n) % 10) / 10,
        pred_excluded_probability=((np.arange(n) + 3) % 10) / 10,
    )


def spine_runs(fig):
    return [trace for trace in fig.data if str(trace.name).startswith("Cycle ")]


def ids_in(trace):
    return [int(value[0]) for value in trace.customdata]


def test_main_style_segments_at_exclusions_and_division_boundaries():
    data = frame()
    expected_ids = [[BIG, BIG + 1, BIG + 2, BIG + 3], [BIG + 4], [BIG + 7, BIG + 8],
                    [BIG + 9, BIG + 10, BIG + 11]]
    expected_x = [[0.0, 0.1, 0.2, 0.3], [0.4], [0.7, 0.8], [0.9, 1.0, 1.1]]
    fig = build_trajectory_plot(data)
    runs = spine_runs(fig)
    assert [ids_in(trace) for trace in runs] == expected_ids
    for trace, expected in zip(runs, expected_x):
        np.testing.assert_allclose(trace.x, expected)
    # Exclusion gaps do not advance the cycle; division boundaries advance it.
    expected_groups = [0, 1, 1, 2]
    assert [trace.line.color for trace in runs] == [
        TRAJECTORY_TRACE_COLORS[group % len(TRAJECTORY_TRACE_COLORS)] for group in expected_groups]
    # No excluded observation is connected; division rows terminate their run.
    assert all(ids_in(trace) != [BIG + 5] and BIG + 5 not in ids_in(trace) and BIG + 6 not in ids_in(trace)
               for trace in runs)
    assert all(trace.mode == "markers+lines" and trace.showlegend is False for trace in runs)
    applied = next(trace for trace in fig.data if trace.name == "Applied exclusion")
    divisions = next(trace for trace in fig.data if trace.name == "Division label")
    assert ids_in(applied) == [BIG + 5, BIG + 6]
    assert ids_in(divisions) == [BIG + 3, BIG + 8]
    assert list(applied.y) == [data.spine_length.iloc[5], data.spine_length.iloc[6]]  # measured positions
    assert applied.marker.symbol == APPLIED_EXCLUSION_MARKER["symbol"]
    assert applied.marker.size == APPLIED_EXCLUSION_MARKER["size"]
    assert applied.marker.color == APPLIED_EXCLUSION_MARKER["color"]
    assert divisions.marker.symbol == DIVISION_LABEL_MARKER["symbol"]
    assert divisions.marker.color == DIVISION_LABEL_MARKER["color"]


def test_main_layout_log_unified_lasso_revision_and_margins():
    fig = build_trajectory_plot(frame(), plot_epoch=7)
    assert fig.layout.yaxis.type == "log"
    assert fig.layout.yaxis.title.text == "Spine length (µm)"
    assert fig.layout.hovermode == "x unified"
    assert fig.layout.dragmode == "lasso"
    assert fig.layout.selectionrevision == 7
    assert fig.layout.uirevision == "a"
    assert fig.layout.height == 600
    assert fig.layout.margin.l == 70 and fig.layout.margin.r == 25
    annotations = {item.text for item in fig.layout.annotations}
    assert SPINE_SUPLOT_TITLE in annotations
    assert fig.layout.title.text == "a"


def test_palette_cycles_across_many_boundary_runs():
    n = 50
    data = pd.DataFrame({
        "trajectory_key": ["cycle"] * n,
        "fov_cell_index": pd.Series(np.arange(BIG, BIG + n), dtype="int64"),
        "time_h": np.arange(n, dtype=float),
        "spine_length": np.linspace(2, 6, n),
        "division_point": [i % 2 == 1 for i in range(n)],
        "excluded_point": [False] * n,
    })
    runs = spine_runs(build_trajectory_plot(data))
    groups = [int(trace.name.split()[-1]) - 1 for trace in runs]
    assert groups == list(range(25))  # 25 division-terminated runs; palette repeats after 16
    assert [trace.line.color for trace in runs] == [
        TRAJECTORY_TRACE_COLORS[group % len(TRAJECTORY_TRACE_COLORS)] for group in groups]
    assert len(TRAJECTORY_TRACE_COLORS) == 16


def test_ai_panel_markers_thresholds_and_unified_probability_runs():
    data = frame()
    pred = probabilities(data)
    before = (data.copy(deep=True), pred.copy(deep=True))
    fig = build_trajectory_plot(data, pred, [0.6, 0.4], plot_epoch=3)
    pd.testing.assert_frame_equal(data, before[0])
    pd.testing.assert_frame_equal(pred, before[1])
    assert fig.layout.yaxis2.range == (0, 1)
    assert fig.layout.yaxis2.title.text == "Probability"
    assert fig.layout.height == 900
    assert {item.text for item in fig.layout.annotations} >= {SPINE_SUPLOT_TITLE, AI_SUPLOT_TITLE}
    probability_traces = [trace for trace in fig.data
                          if trace.name in {"Division probability", "Exclusion probability"}]
    assert len(probability_traces) == 8  # four visible runs x two labels
    assert {trace.yaxis for trace in probability_traces} == {"y2"}
    assert all(trace.x is not None and len(trace.x) > 0 for trace in probability_traces)
    assert [len(trace.x) for trace in probability_traces[::2]] == [4, 1, 2, 3]
    division = next(trace for trace in fig.data if trace.name == "AI division")
    exclusion = next(trace for trace in fig.data if trace.name == "AI exclusion")
    # threshold_pair stores float32 thresholds, matching predict_dataframe.
    expected_division = [i for i in range(len(data))
                         if pred.pred_division_probability.iloc[i] >= np.float32(0.6)]
    expected_exclusion = [i for i in range(len(data))
                          if pred.pred_excluded_probability.iloc[i] >= np.float32(0.4)]
    assert ids_in(division) == [int(data.fov_cell_index.iloc[i]) for i in expected_division]
    assert ids_in(exclusion) == [int(data.fov_cell_index.iloc[i]) for i in expected_exclusion]
    assert division.marker.symbol == AI_DIVISION_MARKER["symbol"]
    assert division.marker.color == AI_DIVISION_MARKER["color"]
    assert exclusion.marker.symbol == AI_EXCLUSION_MARKER["symbol"]
    assert exclusion.marker.color == AI_EXCLUSION_MARKER["color"]
    assert len(fig.layout.shapes) == 2
    assert {shape.line.dash for shape in fig.layout.shapes} == {"dot", "dash"}
    threshold_labels = [item.text for item in fig.layout.annotations if "threshold" in str(item.text)]
    assert len(threshold_labels) == 2 and any("0.60" in text for text in threshold_labels)


def test_probabilities_without_thresholds_have_no_ai_markers_or_threshold_labels():
    data = frame()
    fig = build_trajectory_plot(data, probabilities(data))
    assert fig.layout.yaxis2.range == (0, 1)
    assert not any(trace.name in {"AI division", "AI exclusion"} for trace in fig.data)
    assert len(fig.layout.shapes) == 0
    assert not any("threshold" in str(item.text).lower() for item in fig.layout.annotations)
    assert not any(trace.name.startswith("AI") for trace in fig.data)


def test_thresholds_without_predictions_do_not_create_ai_panel_or_markers():
    fig = build_trajectory_plot(frame(), thresholds=[0.6, 0.4])
    assert "yaxis2" not in fig.to_plotly_json()["layout"]
    assert not any(trace.name.startswith("AI") for trace in fig.data)
    assert len(fig.layout.shapes) == 0


def test_original_labels_overlay_aligns_by_ids_and_uses_semantic_markers():
    data = frame()
    original = frame().iloc[::-1].reset_index(drop=True)
    original.loc[original.fov_cell_index == BIG, "division_point"] = True
    original.loc[original.fov_cell_index == BIG + 9, "excluded_point"] = True
    fig = build_trajectory_plot(data, original_labels=original)
    div = next(trace for trace in fig.data if trace.name == "Original division")
    exc = next(trace for trace in fig.data if trace.name == "Original exclusion")
    assert ids_in(div) == [BIG, BIG + 3, BIG + 8]
    assert ids_in(exc) == [BIG + 5, BIG + 6, BIG + 9]
    assert div.marker.symbol == ORIGINAL_DIVISION_MARKER["symbol"]
    assert exc.marker.symbol == ORIGINAL_EXCLUSION_MARKER["symbol"]
    assert all(isinstance(value[0], str) for trace in fig.data for value in trace.customdata)


def test_overlay_id_and_measurement_mismatches_reject_before_drawing():
    data = frame()
    pred = probabilities(data)
    with pytest.raises(ValueError, match="exactly"):
        build_trajectory_plot(data, pred.iloc[:-1])
    extra = pd.concat([pred, pred.iloc[[0]].assign(fov_cell_index=BIG + 999)], ignore_index=True)
    with pytest.raises(ValueError, match="exactly"):
        build_trajectory_plot(data, extra)
    missing = pred.copy(); missing["fov_cell_index"] = [
        (BIG + 999 if value == BIG + 5 else value) for value in missing.fov_cell_index]
    with pytest.raises(ValueError, match="exactly"):
        build_trajectory_plot(data, missing)
    shifted = pred.copy(); shifted.loc[0, "time_h"] += 1
    with pytest.raises(ValueError, match="align"):
        build_trajectory_plot(data, shifted)
    bad_original = frame().copy(); bad_original.loc[0, "spine_length"] += 1
    with pytest.raises(ValueError, match="align"):
        build_trajectory_plot(data, original_labels=bad_original)
    with pytest.raises(ValueError, match="one trajectory"):
        build_trajectory_plot(pd.concat([data, data.assign(trajectory_key="b")], ignore_index=True))
    invalid = pred.copy(); invalid.loc[0, "pred_division_probability"] = np.nan
    with pytest.raises(ValueError, match="probabilit"):
        build_trajectory_plot(data, invalid)
    nonpositive = data.copy(); nonpositive.loc[0, "spine_length"] = 0
    with pytest.raises(ValueError, match="positive"):
        build_trajectory_plot(nonpositive)


def test_exact_int64_ids_and_selection_never_use_float_positions():
    data = frame()
    fig = build_trajectory_plot(data, probabilities(data), [0.6, 0.4])
    assert str(BIG) in fig.to_json()
    for trace in fig.data:
        for value in trace.customdata:
            assert isinstance(value[0], str)
            assert int(value[0]) in set(data.fov_cell_index)
    first = fig.data[0].customdata[0]
    points = [{"point_index": 999, "customdata": first}, {"customdata": first},
              {"point_index": 0}]
    assert selected_observation_ids({"selection": {"points": points}}, data.fov_cell_index) == [BIG]
    for payload in [float(BIG), str(BIG) + ".0", True, "bad", str(BIG + 999)]:
        with pytest.raises(ValueError):
            selected_observation_ids({"points": [{"customdata": payload}]}, data.fov_cell_index)
    assert selected_observation_ids(None, data.fov_cell_index) == []
    assert selected_observation_ids({"selection": {"points": []}}, data.fov_cell_index) == []


@pytest.mark.parametrize("epoch", [-1, 1.5, True, "2", None])
def test_plot_epoch_must_be_nonnegative_integer(epoch):
    with pytest.raises(ValueError, match="plot_epoch"):
        build_trajectory_plot(frame(), plot_epoch=epoch)


def test_backward_compatible_positional_api_and_default_revision():
    data = frame(); pred = probabilities(data)
    fig = build_trajectory_plot(data, pred, [0.6, 0.4], frame())
    assert fig.layout.selectionrevision == 0
    assert next(trace for trace in fig.data if trace.name == "Division probability")
    assert next(trace for trace in fig.data if trace.name == "Original division")


def test_reusing_figure_replaces_only_traces_and_never_accumulates_thresholds():
    data = frame()
    first = build_trajectory_plot(data, probabilities(data), [0.6, 0.4])
    layout = first.layout
    assert len(first.layout.shapes) == 2
    assert len(first.layout.annotations) == 4  # two subtitles + two thresholds
    labels = data.copy()
    labels.loc[3, "division_point"] = True
    second = build_trajectory_plot(labels, probabilities(labels), [0.6, 0.4],
                                   plot_epoch=1, figure=first)
    assert second is first and second.layout is layout
    assert len(second.layout.shapes) == 2
    assert len(second.layout.annotations) == 4
    assert any(trace.name == "Division label" for trace in second.data)
    assert second.layout.selectionrevision == 1
    without = build_trajectory_plot(labels, figure=second)
    assert without is not second
    assert len(without.layout.annotations) == 1
