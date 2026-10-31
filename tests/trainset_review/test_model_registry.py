"""Shared released model catalogue, package provenance, and Streamlit selection."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moma_lineage_analysis import model_registry
from moma_lineage_analysis.paths import PACKAGE_MODELS_RUN_DIR, PACKAGE_ROOT
from moma_lineage_analysis.trajectory_cleaning.training import load_checkpoint, predict_dataframe


LEGACY_REGISTRY = {
    "R019 CNN (reviewed corpus · production)": (
        "best_model.pt",
        "Current production model (promoted 2026-10-31): CNN trained on the "
        "user-reviewed five-batch corpus (1,152 trajectories / 385,815 "
        "observations, eight sources). Test division-event F1 0.997, exclusion "
        "F1 0.798, exclusion AP 0.878. Validation-only rule winner (mean F1 "
        "0.901); descriptive same-row panels versus the previous R010 CNN "
        "favoured it on the small clean common panel, without significance "
        "testing.",
    ),
    "R002 CNN (mixed corpus · all conditions)": (
        "verified_20260916/cnn/best_model.pt",
        "Trained on all conditions incl. the 21.6 h RDM-Gluc dataset. Safe "
        "default for mixed workloads: tail division recall 0.99, last-event "
        "miss rate <1%; slightly lower precision on short trajectories.",
    ),
    "R002 ResNet (mixed corpus · all conditions)": (
        "verified_20260916/resnet/best_model.pt",
        "Mixed-corpus dilated ResNet; same scope as R002 CNN with a wider "
        "receptive field (121 frames).",
    ),
    "R002 BiLSTM (mixed corpus · all conditions)": (
        "verified_20260916/lstm/best_model.pt",
        "Mixed-corpus bidirectional LSTM; best last-division miss rate on "
        "the mixed test split (0.0%), slower to train/predict.",
    ),
    "R003 CNN (RDM-Gluc only · long experiments)": (
        "rdm_only_20260916/cnn/best_model.pt",
        "Specialist for long RDM-Gluc-like experiments: best exclusion "
        "detection (F1 0.85) and tail recall 0.99. Do NOT use for short "
        "≤10 h trajectories from other conditions.",
    ),
    "R003 ResNet (RDM-Gluc only · long experiments)": (
        "rdm_only_20260916/resnet/best_model.pt",
        "RDM-only specialist; zero last-division misses on the RDM-Gluc "
        "test split. Not for other conditions.",
    ),
    "R003 BiLSTM (RDM-Gluc only · long experiments)": (
        "rdm_only_20260916/lstm/best_model.pt",
        "RDM-only specialist; balanced division/exclusion behavior on long "
        "experiments. Not for other conditions.",
    ),
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_existing_options_and_production_default_are_preserved():
    registry = model_registry.AI_MODEL_REGISTRY
    assert len(registry) == 8
    assert next(iter(registry)) == model_registry.DEFAULT_AI_MODEL_TAG
    assert model_registry.DEFAULT_AI_CHECKPOINT == str(PACKAGE_MODELS_RUN_DIR / "best_model.pt")
    for tag, (relative, guidance) in LEGACY_REGISTRY.items():
        assert registry[tag] == (str(PACKAGE_MODELS_RUN_DIR / relative), guidance)
    assert _sha256(model_registry.DEFAULT_AI_CHECKPOINT) == (
        "816a8b392ec206dca578a093c6cbc2fc9ec5ba12480d6aca7c8b08349796af99"
    )
    assert [tag for tag, details in model_registry.MODEL_DETAILS.items()
            if details["is_production"]] == [model_registry.DEFAULT_AI_MODEL_TAG]


@pytest.mark.parametrize("tag", list(model_registry.AI_MODEL_REGISTRY))
def test_all_registered_paths_and_companion_files_exist_with_pinned_hashes(tag):
    checkpoint, guidance = model_registry.AI_MODEL_REGISTRY[tag]
    details = model_registry.MODEL_DETAILS[tag]
    assert isinstance(checkpoint, str) and isinstance(guidance, str) and guidance
    assert isinstance(model_registry.AI_MODEL_REGISTRY[tag], tuple)
    assert details["checkpoint"] == checkpoint
    assert _sha256(checkpoint) == details["sha256"]
    for path in (checkpoint, details["metrics_path"], details["splits_path"]):
        assert Path(path).is_absolute()
        assert Path(path).is_file()
        assert Path(path).is_relative_to(PACKAGE_MODELS_RUN_DIR)


def test_r015_package_is_exact_run_copy_and_manifest_pins_all_artifacts():
    details = model_registry.MODEL_DETAILS[model_registry.R015_CNN_TAG]
    assert details["is_production"] is False
    assert details["rows"] == 399168 and details["trajectories"] == 1197
    assert details["receptive_field"] == 17
    assert details["sha256"] == "9753298022805b4bc9f277d70e4da29aa8d88cac84a9575691477f332ff21687"
    release_path = Path(details["release_manifest_path"])
    manifest = json.loads(release_path.read_text(encoding="utf-8"))
    assert manifest["registry_tag"] == model_registry.R015_CNN_TAG
    assert manifest["is_production"] is False
    # The R015 release manifest records the production checkpoint at release
    # time (R010 CNN), not the current R019 production pin.
    assert manifest["production_sha256"] == (
        "6e7c5a5896c90bd0de2b068af087f2d9c942b2de94e8deb87e419fd005cb230b"
    )
    assert manifest["dataset"]["lineage_id"] == details["lineage_id"]
    assert manifest["dataset"]["rows"] == 399168
    assert manifest["dataset"]["trajectories"] == 1197
    assert set(manifest["artifacts"]) == {"best_model.pt", "history.csv", "metrics.json", "splits.json"}
    root = PACKAGE_ROOT.parents[1]
    for filename, record in manifest["artifacts"].items():
        packaged = release_path.parent / filename
        assert record["source"] == f"runs/merged_20261029/cnn/{filename}"
        assert packaged.stat().st_size == record["bytes"]
        assert _sha256(packaged) == record["sha256"]
        source = root / record["source"]
        if source.is_file():
            assert source.read_bytes() == packaged.read_bytes()


def test_registry_resolves_and_r015_predicts_from_independent_working_directory(tmp_path, monkeypatch):
    original = dict(model_registry.AI_MODEL_REGISTRY)
    monkeypatch.chdir(tmp_path)
    registry = importlib.reload(model_registry)
    assert registry.AI_MODEL_REGISTRY == original
    details = registry.MODEL_DETAILS[registry.R015_CNN_TAG]
    model, normalizer, thresholds, metadata = load_checkpoint(details["checkpoint"], device="cpu")
    metrics = json.loads(Path(details["metrics_path"]).read_text(encoding="utf-8"))
    np.testing.assert_array_equal(
        thresholds, np.array([metrics["thresholds"][label]
                              for label in ("division_point", "excluded_point")], dtype=np.float32),
    )
    np.testing.assert_allclose(thresholds, [0.74, 0.87], rtol=0, atol=1e-7)
    assert metadata["model_config"]["model_type"] == "cnn"
    assert model.receptive_field == 17
    assert metadata["training_data"]["rows"] == 399168
    assert metadata["training_data"]["trajectories"] == 1197
    assert metadata["dataset_lineage"]["composition"]["lineage_ids"] == [details["lineage_id"]]
    frame = pd.DataFrame({
        "trajectory_key": ["synthetic"] * 24,
        "fov_cell_index": np.arange(24),
        "time_h": np.arange(24) / 15,
        "spine_length": 2 + (np.arange(24) % 6) * 0.2,
    })
    result = predict_dataframe(model, frame, normalizer=normalizer, thresholds=thresholds, device="cpu")
    assert len(result) == len(frame)
    for label in ("division", "excluded"):
        probabilities = result[f"pred_{label}_probability"]
        assert np.isfinite(probabilities).all()
        assert probabilities.between(0, 1).all()
    assert result["fov_cell_index"].tolist() == frame["fov_cell_index"].tolist()


def test_existing_app_imports_shared_catalogue_instead_of_redefining_it():
    tree = ast.parse((PACKAGE_ROOT / "app/Moma_lineage_streamlit.py").read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.ImportFrom) and node.module == "moma_lineage_analysis.model_registry"
        and any(name.name == "AI_MODEL_REGISTRY" for name in node.names)
        for node in tree.body
    )
    assert not any(
        isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "AI_MODEL_REGISTRY"
            for target in node.targets
        ) for node in tree.body
    )


def _element(elements, label):
    return next(element for element in elements if element.label == label)


def test_existing_ai_panel_can_select_and_load_r015_on_cpu(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("MOMA_AI_MODEL_PATH", raising=False)
    app = AppTest.from_file(
        str(PACKAGE_ROOT / "app/Moma_lineage_streamlit.py"), default_timeout=30,
    ).run()
    assert not app.exception
    _element(app.toggle, "Enable AI-assisted cleaning").set_value(True)
    app.run()
    selector = _element(app.selectbox, "Model checkpoint")
    assert selector.value == model_registry.DEFAULT_AI_MODEL_TAG
    assert model_registry.R015_CNN_TAG in selector.options
    selector.set_value(model_registry.R015_CNN_TAG)
    app.run()
    assert not app.exception
    assert app.session_state["ai_checkpoint_path"] == model_registry.AI_MODEL_REGISTRY[model_registry.R015_CNN_TAG][0]
    _element(app.selectbox, "Device").set_value("cpu")
    app.run()
    _element(app.button, "📦 Load / Reload Model").click()
    app.run()
    assert not app.exception
    assert app.session_state["ai_model_metadata"]["training_data"]["trajectories"] == 1197
    assert _element(app.slider, "Division threshold").value == pytest.approx(0.74, abs=1e-7)
    assert _element(app.slider, "Noise threshold").value == pytest.approx(0.87, abs=1e-7)
    assert _sha256(model_registry.DEFAULT_AI_CHECKPOINT) == model_registry.PRODUCTION_SHA256
