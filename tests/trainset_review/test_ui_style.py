"""Shared trajectory style and keyboard contract for both apps."""
import json
import re
from pathlib import Path

import pytest
from moma_lineage_analysis.trajectory_view_style import (
    AI_DIVISION_MARKER, APPLIED_EXCLUSION_MARKER, TRAJECTORY_TRACE_COLORS,
    shortcut_script, trajectory_plot_runs,
)
from moma_lineage_analysis.paths import PACKAGE_ROOT


def test_palette_and_markers_are_shared_high_contrast_values():
    assert len(TRAJECTORY_TRACE_COLORS) == 16
    assert len(set(TRAJECTORY_TRACE_COLORS)) == 16
    assert APPLIED_EXCLUSION_MARKER["symbol"] == "x"
    assert AI_DIVISION_MARKER["symbol"].endswith("open")


def test_runs_split_at_exclusions_and_division_boundaries():
    frame = __import__("pandas").DataFrame({
        "fov_cell_index": [1, 2, 3, 4, 5, 6],
        "time_h": [0., 1., 2., 3., 4., 5.],
    })
    runs = trajectory_plot_runs(frame, [3], [2])
    assert [run[1].fov_cell_index.tolist() for run in runs] == [[1, 2], [4, 5, 6]]
    assert [run[0] for run in runs] == [0, 1]
    assert trajectory_plot_runs(frame, [1, 2], [])[0][1].fov_cell_index.tolist() == [3, 4, 5, 6]


def test_shortcut_script_guards_inputs_dialogs_and_escapes_labels():
    script = shortcut_script({"ArrowLeft": "Prev", "d": "Mark <division>", "e": "Exclude & restore"})
    assert "__momaTrajectoryKeyHandler" in script
    assert "ArrowLeft" in script and "Mark <division>" in script
    assert "role=\"dialog\"" in script and "role=\"slider\"" in script
    assert "role=\"grid\"" in script and "role=\"combobox\"" in script
    assert "stDataEditor" in script and "stDataFrame" in script
    assert "'INPUT', 'TEXTAREA', 'SELECT'" in script
    assert "target.closest" in script
    assert "e.ctrlKey" in script and "e.metaKey" in script and "e.altKey" in script
    assert "target.isContentEditable" in script and "e.isComposing" in script
    assert "visible(button)" in script and "e.preventDefault()" in script
    assert "button.disabled" in script
    assert "<\\/script" not in script
    with pytest.raises(ValueError):
        shortcut_script({})


def test_review_app_has_no_dataset_export_or_manual_save_controls():
    review = (PACKAGE_ROOT / "trainset_review/streamlit_app.py").read_text(encoding="utf-8")
    for forbidden in ('Finalize / export', 'Finalize/export', 'Prepare current dataset export',
                      'Download dataset CSV', 'Save table labels', 'export_snapshot'):
        assert forbidden not in review
    assert "st.tabs(['Review trajectories', 'Evaluate dataset'])" in review
    assert "review.table_labels(frame, edited)" in review
    # Label edits and deletion must rerun only the review workspace, not the page.
    assert '@st.fragment\ndef review_workspace():' in review
    assert 'review_workspace()' in review
    # Save feedback is a transient bottom-left toast, never a layout banner/path.
    assert "st.toast(st.session_state.pop('review_flash'), icon='✅', duration=3)" in review
    assert '[data-testid="stToastContainer"]' in review
    assert 'Snapshot:' not in review
    assert 'Selected observation IDs' not in review


def test_both_apps_use_the_shared_handler_and_palette():
    main = (PACKAGE_ROOT / "app/Moma_lineage_streamlit.py").read_text(encoding="utf-8")
    review = (PACKAGE_ROOT / "trainset_review/streamlit_app.py").read_text(encoding="utf-8")
    assert "shortcut_script(" in main and "shortcut_script(" in review
    assert "__momaLineageKeyHandler" not in main  # one shared handler only
    assert "TRAJECTORY_TRACE_COLORS = (" not in main
    assert "from moma_lineage_analysis.trajectory_view_style import" in review
