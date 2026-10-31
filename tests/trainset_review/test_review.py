"""Pure review/Plotly regressions, including opaque int64 browser identities."""
import json

import numpy as np
import pandas as pd
import pytest

from moma_lineage_analysis.trainset_review.review import (
    edit_labels, resolve_labels, table_labels, validate_frame,
)
from moma_lineage_analysis.trainset_review.plots import selected_observation_ids

BIG = 2**53 + 11


def frame():
    return pd.DataFrame({"trajectory_key": ["a"]*4,
        "fov_cell_index": pd.Series([BIG, BIG+1, BIG+2, BIG+3], dtype="int64"),
        "time_h": [0., 1., 2., 3.], "spine_length": [4., 20., 18., 2.96],
        "division_point": [False, True, True, False], "excluded_point": [False, True, True, False]})


@pytest.mark.parametrize("action,division,excluded", [
    ("add_division", True, False), ("remove_division", False, False),
    ("exclude", False, True), ("restore", False, False), ("clear", False, False)])
def test_edit_is_explicit_copy_preserving_measurements_and_independent_labels(action, division, excluded):
    data = frame(); before = data.copy(deep=True)
    result = edit_labels(data, [BIG], action)
    assert result.division_point.iloc[0] == division
    assert result.excluded_point.iloc[0] == excluded
    pd.testing.assert_frame_equal(data, before)
    pd.testing.assert_frame_equal(result.drop(columns=["division_point", "excluded_point"]),
                                  before.drop(columns=["division_point", "excluded_point"]))
    # Merely editing another row never resolves the existing overlapping run.
    assert result.division_point.iloc[1:3].all() and result.excluded_point.iloc[1:3].all()


def test_restore_does_not_remove_division_and_exclude_does_not_remove_division():
    data = frame()
    assert edit_labels(data, [BIG+1], "restore").division_point.iloc[1]
    assert edit_labels(data, [BIG+1], "exclude").division_point.iloc[1]
    assert not edit_labels(data, [BIG+1], "clear").loc[1, ["division_point", "excluded_point"]].any()


@pytest.mark.parametrize("ids,action", [([BIG+100], "clear"), ([float(BIG)], "clear"),
    ([True], "clear"), ([str(BIG)], "clear"), ([BIG], "invent")])
def test_edit_rejects_invalid_or_unknown_ids_and_actions(ids, action):
    with pytest.raises(ValueError): edit_labels(frame(), ids, action)


def test_validate_boolean_finite_positive_duplicate_ids_and_empty():
    for column, value in [("division_point", 1), ("excluded_point", "False"),
                           ("time_h", np.inf), ("spine_length", np.nan), ("spine_length", 0)]:
        data = frame(); data[column] = data[column].astype(object); data.loc[0, column] = value
        with pytest.raises(ValueError): validate_frame(data)
    data = frame(); data.loc[1, "fov_cell_index"] = BIG
    with pytest.raises(ValueError, match="Duplicate"): validate_frame(data)
    with pytest.raises(ValueError, match="empty"): validate_frame(frame().iloc[:0])
    data = pd.concat([frame(), frame().assign(trajectory_key="b")], ignore_index=True)
    validate_frame(data)  # FOV IDs can repeat across distinct trajectories.
    with pytest.raises(ValueError, match="ambiguous"): edit_labels(data, [BIG], "clear")


def test_resolve_uses_packaged_policy_preserves_int64_audit_and_input_order():
    data = frame().iloc[[3, 1, 0, 2]].copy(); data.index = [4, 4, 2, 9]; before = data.copy(deep=True)
    result = resolve_labels(data)
    assert set(result) == {"frame", "decisions", "status_counts", "removed", "added"}
    assert result["frame"].loc[result["frame"].division_point, "fov_cell_index"].tolist() == [BIG]
    assert result["removed"] == 2 and result["added"] == 1
    assert result["status_counts"] == {"unchanged": 0, "coalesced_previous": 1,
                                        "shifted_previous": 1, "needs_review": 0}
    audit = result["decisions"]
    assert audit.source_index.tolist() == [BIG+1, BIG+2]
    assert audit.effective_index.tolist() == [BIG, BIG]
    assert all(isinstance(value, int) for value in audit.source_index)
    assert result["frame"].index.equals(before.index)
    pd.testing.assert_frame_equal(data, before)
    assert result["frame"].excluded_point.equals(before.excluded_point)


def test_resolve_separates_trajectories_with_repeated_fov_ids():
    data = pd.concat([frame(), frame().assign(trajectory_key="b")], ignore_index=True)
    result = resolve_labels(data)
    assert result["removed"] == 4 and result["added"] == 2
    assert result["decisions"].trajectory_key.tolist() == ["a", "a", "b", "b"]
    assert result["frame"].groupby("trajectory_key").division_point.sum().to_dict() == {"a": 1, "b": 1}


def test_selection_exact_decimal_strings_and_no_position_fallback():
    event = {"selection": {"points": [{"point_index": 999, "customdata": [str(BIG)]},
            {"customdata": str(BIG+1)}, {"customdata": [str(BIG)]}, {"point_index": 0}]}}
    assert selected_observation_ids(event, [BIG, BIG+1]) == [BIG, BIG+1]
    assert selected_observation_ids(None, [BIG]) == []
    assert selected_observation_ids({"selection": {"points": []}}, [BIG]) == []
    assert selected_observation_ids({"points": [{"customdata": [BIG]}]}, [BIG]) == [BIG]


@pytest.mark.parametrize("payload", [float(BIG), str(BIG)+".0", str(BIG)+"e0", True,
                                       "01", "bad", str(BIG+100), [str(BIG), "extra"]])
def test_selection_rejects_lossy_malformed_or_unknown_ids(payload):
    with pytest.raises(ValueError):
        selected_observation_ids({"points": [{"customdata": payload}]}, [BIG])


def test_table_labels_autosave_requires_exact_ids_and_boolean_labels():
    data = frame()
    edited = data[["fov_cell_index", "division_point", "excluded_point"]].copy()
    edited["fov_cell_index"] = [str(value) for value in edited.fov_cell_index]
    edited.loc[0, "division_point"] = True
    result = table_labels(data, edited)
    assert result.division_point.iloc[0]
    assert result.excluded_point.equals(data.excluded_point)
    assert data.division_point.equals(frame().division_point)  # input untouched
    with pytest.raises(ValueError, match="observation IDs"):
        table_labels(data, edited.iloc[::-1])
    with pytest.raises(ValueError, match="number"):
        table_labels(data, edited.iloc[:2])
    bad = edited.astype({"division_point": object})
    bad.loc[0, "division_point"] = 1
    with pytest.raises(ValueError, match="boolean"):
        table_labels(data, bad)

