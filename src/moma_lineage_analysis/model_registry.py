"""Shared catalogue of package-shipped trajectory-cleaning checkpoints.

Registry paths are absolute and independent of the caller's working directory.
R019 remains the production default; selecting R015 does not promote it.
"""

from pathlib import Path

from .paths import PACKAGE_MODELS_RUN_DIR


DEFAULT_AI_MODEL_TAG = "R019 CNN (reviewed corpus · production)"
R015_CNN_TAG = "R015 CNN (expanded merged corpus · selectable)"
PRODUCTION_SHA256 = "816a8b392ec206dca578a093c6cbc2fc9ec5ba12480d6aca7c8b08349796af99"
R015_CNN_SHA256 = "9753298022805b4bc9f277d70e4da29aa8d88cac84a9575691477f332ff21687"

# Public compatibility contract: display tag -> (checkpoint path, usage guidance).
AI_MODEL_REGISTRY = {
    "R019 CNN (reviewed corpus · production)": (
        str(PACKAGE_MODELS_RUN_DIR / "best_model.pt"),
        "Current production model (promoted 2026-10-31): CNN trained on the "
        "user-reviewed five-batch corpus (1,152 trajectories / 385,815 "
        "observations, eight sources). Test division-event F1 0.997, exclusion "
        "F1 0.798, exclusion AP 0.878. Validation-only rule winner (mean F1 "
        "0.901); descriptive same-row panels versus the previous R010 CNN "
        "favoured it on the small clean common panel, without significance "
        "testing.",
    ),
    R015_CNN_TAG: (
        str(PACKAGE_MODELS_RUN_DIR / "merged_20261029" / "cnn" / "best_model.pt"),
        "R015 CNN trained on the expanded merged corpus (1,197 trajectories, "
        "399,168 observations, including the 2026-10-29 curated batch); "
        "17-frame receptive field. Available for human-reviewed previews. "
        "R019 CNN is production; paired evaluation of R015 did not support "
        "replacing the earlier R010 CNN.",
    ),
    "R002 CNN (mixed corpus · all conditions)": (
        str(PACKAGE_MODELS_RUN_DIR / "verified_20260916" / "cnn" / "best_model.pt"),
        "Trained on all conditions incl. the 21.6 h RDM-Gluc dataset. Safe "
        "default for mixed workloads: tail division recall 0.99, last-event "
        "miss rate <1%; slightly lower precision on short trajectories.",
    ),
    "R002 ResNet (mixed corpus · all conditions)": (
        str(PACKAGE_MODELS_RUN_DIR / "verified_20260916" / "resnet" / "best_model.pt"),
        "Mixed-corpus dilated ResNet; same scope as R002 CNN with a wider "
        "receptive field (121 frames).",
    ),
    "R002 BiLSTM (mixed corpus · all conditions)": (
        str(PACKAGE_MODELS_RUN_DIR / "verified_20260916" / "lstm" / "best_model.pt"),
        "Mixed-corpus bidirectional LSTM; best last-division miss rate on "
        "the mixed test split (0.0%), slower to train/predict.",
    ),
    "R003 CNN (RDM-Gluc only · long experiments)": (
        str(PACKAGE_MODELS_RUN_DIR / "rdm_only_20260916" / "cnn" / "best_model.pt"),
        "Specialist for long RDM-Gluc-like experiments: best exclusion "
        "detection (F1 0.85) and tail recall 0.99. Do NOT use for short "
        "≤10 h trajectories from other conditions.",
    ),
    "R003 ResNet (RDM-Gluc only · long experiments)": (
        str(PACKAGE_MODELS_RUN_DIR / "rdm_only_20260916" / "resnet" / "best_model.pt"),
        "RDM-only specialist; zero last-division misses on the RDM-Gluc "
        "test split. Not for other conditions.",
    ),
    "R003 BiLSTM (RDM-Gluc only · long experiments)": (
        str(PACKAGE_MODELS_RUN_DIR / "rdm_only_20260916" / "lstm" / "best_model.pt"),
        "RDM-only specialist; balanced division/exclusion behavior on long "
        "experiments. Not for other conditions.",
    ),
}

DEFAULT_AI_CHECKPOINT = AI_MODEL_REGISTRY[DEFAULT_AI_MODEL_TAG][0]

# Pins are recorded from the released artifacts, never inferred from test scores.
_CHECKPOINT_SHA256 = {
    "best_model.pt": PRODUCTION_SHA256,
    "merged_20261029/cnn/best_model.pt": R015_CNN_SHA256,
    "verified_20260916/cnn/best_model.pt": "56ee4e53c7eaca4e4fa991f7b1b22508ff7cf61100795de19db5b9e601ae7750",
    "verified_20260916/resnet/best_model.pt": "e6be1ee05264f8e52a177cc58e3ac14619b841ac7a189e431e0cd0104d5df39d",
    "verified_20260916/lstm/best_model.pt": "909c302edd75d33b69cc229da0b7727a45793d108e282c63625444ce612d227d",
    "rdm_only_20260916/cnn/best_model.pt": "d6e4b036ebaabb58d39bec68bbc81cede2e5eb05fe3b52df98ec15a850e6fa93",
    "rdm_only_20260916/resnet/best_model.pt": "e3e258e13fcbebae4c1cf3508d1de8938f82882b41d5d8f90616767f50875d18",
    "rdm_only_20260916/lstm/best_model.pt": "48433c35dc8bb1400925e79a7011f3307d4c8cb66806cdb4daaba7329592c42e",
}

MODEL_DETAILS = {
    tag: {
        "checkpoint": checkpoint,
        "metrics_path": str(Path(checkpoint).with_name("metrics.json")),
        "splits_path": str(Path(checkpoint).with_name("splits.json")),
        "sha256": _CHECKPOINT_SHA256[Path(checkpoint).relative_to(PACKAGE_MODELS_RUN_DIR).as_posix()],
        "is_production": tag == DEFAULT_AI_MODEL_TAG,
    }
    for tag, (checkpoint, _guidance) in AI_MODEL_REGISTRY.items()
}
MODEL_DETAILS[DEFAULT_AI_MODEL_TAG].update({
    "round": "R019",
    "promotion_record": str(PACKAGE_MODELS_RUN_DIR / "promotion.json"),
    "run_dir": "runs/reviewed_20261031",
    "rows": 385815,
    "trajectories": 1152,
    "lineage_ids": [
        "reviewed_20261031_verified_20260916_no_longruns@v1",
        "reviewed_20261031_verified_20261013@v1",
        "reviewed_20261031_verified_20261016@v1",
        "reviewed_20261031_verified_20261017@v1",
        "reviewed_20261031_verified_20261029@v1",
    ],
})
MODEL_DETAILS[R015_CNN_TAG].update({
    "release_manifest_path": str(
        PACKAGE_MODELS_RUN_DIR / "merged_20261029" / "cnn" / "release_manifest.json"
    ),
    "lineage_id": "verified_merged_20261029@v1",
    "rows": 399168,
    "trajectories": 1197,
    "receptive_field": 17,
})
