"""Focused tests for trajectory grouping, normalization, and sequence models."""

from __future__ import annotations

import os
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
import torch

from moma_lineage_analysis.trajectory_cleaning import (
    ModelConfig,
    TrainConfig,
    TrajectoryDataset,
    TrajectoryNormalizer,
    build_model,
    load_checkpoint,
    predict_dataframe,
    save_checkpoint,
    split_trajectory_keys,
    train_model,
)
from moma_lineage_analysis.trajectory_cleaning.data import collate_trajectories
from moma_lineage_analysis.trajectory_cleaning.training import (
    MaskedFocalBCEWithLogitsLoss,
    cluster_event_indices,
)


def make_dataframe(trajectory_count: int = 12, point_count: int = 10) -> pd.DataFrame:
    rows = []
    for trajectory_index in range(trajectory_count):
        for point_index in range(point_count):
            cycle_position = point_index % 5
            length = 2.0 + 0.55 * cycle_position + 0.03 * trajectory_index
            rows.append(
                {
                    "trajectory_key": f"trajectory_{trajectory_index}",
                    "time_h": point_index / 15.0,
                    "spine_length": length,
                    "division_point": point_index in {4, 9},
                    "excluded_point": (
                        trajectory_index % 2 == 0 and point_index == 7
                    ),
                    "source_marker": trajectory_index * 100 + point_index,
                }
            )
    return pd.DataFrame(rows)


class NormalizationTests(unittest.TestCase):
    def test_time_units_offsets_and_length_scales_are_invariant(self) -> None:
        normalizer = TrajectoryNormalizer()
        time_h = np.array([0.0, 0.1, 0.2, 0.4, 0.5])
        length = np.array([2.0, 2.5, 3.1, 3.8, 2.1])
        reference = normalizer.transform(time_h, length)
        converted = normalizer.transform(time_h * 3600.0 + 1.7e9, length * 10.0)
        np.testing.assert_allclose(reference, converted, rtol=2e-6, atol=2e-6)
        self.assertAlmostEqual(
            reference[3, 0] - reference[2, 0],
            2.0 * (reference[2, 0] - reference[1, 0]),
            places=6,
        )

    def test_nonpositive_length_warns_but_stays_finite(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            values = TrajectoryNormalizer().transform(
                [0.0, 1.0, 2.0],
                [2.0, 0.0, 3.0],
            )
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(any("Nonpositive spine lengths" in str(item.message) for item in caught))

    def test_duplicate_time_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            TrajectoryNormalizer().transform([0.0, 1.0, 1.0], [2.0, 3.0, 4.0])


class DatasetTests(unittest.TestCase):
    def test_trajectory_key_groups_and_sorts_rows(self) -> None:
        dataframe = make_dataframe(2, 5).sample(frac=1.0, random_state=3).reset_index(drop=True)
        overlap_row = dataframe.index[0]
        dataframe.loc[overlap_row, ["division_point", "excluded_point"]] = True
        overlap_marker = dataframe.loc[overlap_row, "source_marker"]
        dataset = TrajectoryDataset(dataframe)
        self.assertEqual(len(dataset), 2)
        for record in dataset.records:
            ordered_time = dataframe.iloc[record.row_positions]["time_h"].to_numpy()
            self.assertTrue(np.all(np.diff(ordered_time) > 0))
            ordered_markers = dataframe.iloc[record.row_positions]["source_marker"].to_numpy()
            if overlap_marker in ordered_markers:
                position = int(np.flatnonzero(ordered_markers == overlap_marker)[0])
                np.testing.assert_array_equal(
                    record.targets[position].numpy(),
                    np.array([1.0, 1.0], dtype=np.float32),
                )

    def test_split_sets_are_disjoint_and_complete(self) -> None:
        dataframe = make_dataframe(30, 6)
        split = split_trajectory_keys(
            dataframe,
            validation_fraction=0.2,
            test_fraction=0.2,
            seed=9,
        )
        train = set(split["train"])
        validation = set(split["validation"])
        test = set(split["test"])
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(
            train | validation | test,
            set(dataframe["trajectory_key"].unique()),
        )

    def test_source_stratified_split_represents_every_source(self) -> None:
        dataframe = make_dataframe(24, 6)
        trajectory_number = dataframe["trajectory_key"].str.rsplit(
            "_", n=1
        ).str[-1].astype(int)
        dataframe["source_dataset"] = np.where(
            trajectory_number < 12, "source_a", "source_b"
        )
        split = split_trajectory_keys(
            dataframe,
            validation_fraction=0.2,
            test_fraction=0.2,
            seed=9,
            source_column="source_dataset",
        )

        for split_name, keys in split.items():
            with self.subTest(split=split_name):
                represented = set(
                    dataframe.loc[
                        dataframe["trajectory_key"].isin(keys),
                        "source_dataset",
                    ]
                )
                self.assertEqual(represented, {"source_a", "source_b"})
        self.assertEqual(
            {name: len(keys) for name, keys in split.items()},
            {"train": 16, "validation": 4, "test": 4},
        )

    def test_source_stratified_split_rejects_ambiguous_trajectory(self) -> None:
        dataframe = make_dataframe(8, 4)
        dataframe["source_dataset"] = "source_a"
        first_key = dataframe["trajectory_key"].iloc[0]
        first_key_rows = dataframe.index[
            dataframe["trajectory_key"].eq(first_key)
        ]
        dataframe.loc[first_key_rows[-1], "source_dataset"] = "source_b"
        with self.assertRaisesRegex(ValueError, "exactly one source"):
            split_trajectory_keys(
                dataframe,
                validation_fraction=0.2,
                test_fraction=0.2,
                source_column="source_dataset",
            )


class ModelTests(unittest.TestCase):
    def test_all_models_return_two_labels_per_point(self) -> None:
        dataset = TrajectoryDataset(make_dataframe(3, 8))
        batch = collate_trajectories([dataset[0], dataset[1], dataset[2]])
        for model_type in ("cnn", "resnet", "lstm"):
            with self.subTest(model_type=model_type):
                model = build_model(
                    ModelConfig(
                        model_type=model_type,
                        hidden_size=8,
                        num_layers=1,
                        kernel_size=3,
                        dropout=0.0,
                    )
                )
                output = model(batch["features"], batch["mask"])
                self.assertEqual(output.shape, (3, 8, 2))

    def test_padding_does_not_change_masked_loss(self) -> None:
        loss_function = MaskedFocalBCEWithLogitsLoss(torch.ones(2), gamma=0.0)
        logits = torch.zeros(2, 4, 2)
        targets = torch.zeros_like(logits)
        mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
        reference = loss_function(logits, targets, mask)
        modified = logits.clone()
        modified[0, 2:] = 1000.0
        self.assertAlmostEqual(
            float(reference),
            float(loss_function(modified, targets, mask)),
            places=7,
        )

    def test_convolution_predictions_are_invariant_to_batch_padding(self) -> None:
        dataframe = make_dataframe(2, 8)
        extra = make_dataframe(1, 13).assign(trajectory_key="long_trajectory")
        dataset = TrajectoryDataset(pd.concat([dataframe, extra], ignore_index=True))
        short = dataset[0]
        long = next(record for record in dataset.records if record.key == "long_trajectory")
        for model_type in ("cnn", "resnet"):
            with self.subTest(model_type=model_type):
                torch.manual_seed(7)
                model = build_model(
                    ModelConfig(
                        model_type=model_type,
                        hidden_size=8,
                        num_layers=2,
                        kernel_size=3,
                        dropout=0.0,
                    )
                ).eval()
                alone = collate_trajectories([short])
                padded = collate_trajectories([short, long])
                with torch.no_grad():
                    output_alone = model(alone["features"], alone["mask"])[0]
                    output_padded = model(padded["features"], padded["mask"])[0, : short.length]
                torch.testing.assert_close(output_alone, output_padded)

    def test_one_optimization_step_has_finite_gradients(self) -> None:
        dataset = TrajectoryDataset(make_dataframe(4, 8))
        batch = collate_trajectories([dataset[index] for index in range(4)])
        model = build_model(
            ModelConfig(
                model_type="resnet",
                hidden_size=8,
                num_layers=1,
                kernel_size=3,
                dropout=0.0,
            )
        )
        loss_function = MaskedFocalBCEWithLogitsLoss(torch.tensor([2.0, 4.0]), gamma=1.0)
        loss = loss_function(
            model(batch["features"], batch["mask"]),
            batch["targets"],
            batch["mask"],
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_training_loop_smoke(self) -> None:
        dataframe = make_dataframe(12, 10)
        train_dataset = TrajectoryDataset(
            dataframe,
            keys=[f"trajectory_{index}" for index in range(8)],
        )
        validation_dataset = TrajectoryDataset(
            dataframe,
            keys=[f"trajectory_{index}" for index in range(8, 12)],
        )
        model = build_model(
            ModelConfig(
                model_type="cnn",
                hidden_size=8,
                num_layers=1,
                kernel_size=3,
                dropout=0.0,
            )
        )
        trained, history, thresholds = train_model(
            model,
            train_dataset,
            validation_dataset,
            config=TrainConfig(
                epochs=2,
                batch_size=4,
                patience=2,
                device="cpu",
                seed=12,
            ),
        )
        self.assertIs(trained, model)
        self.assertEqual(len(history), 2)
        self.assertEqual(thresholds.shape, (2,))
        self.assertTrue(np.isfinite(thresholds).all())


class InferenceAndEventTests(unittest.TestCase):
    def test_adjacent_event_points_are_clustered(self) -> None:
        positive = np.array([False, True, True, False, True, False])
        excluded = np.array([False, True, False, False, False, False])
        score = np.array([0.0, 0.8, 0.7, 0.0, 0.9, 0.0])
        np.testing.assert_array_equal(
            cluster_event_indices(positive, excluded=excluded),
            np.array([2, 4]),
        )
        np.testing.assert_array_equal(
            cluster_event_indices(positive, score=score),
            np.array([1, 4]),
        )

    def test_checkpoint_and_prediction_preserve_original_row_order(self) -> None:
        dataframe = make_dataframe(3, 6).sample(frac=1.0, random_state=8).reset_index(drop=True)
        original_markers = dataframe["source_marker"].tolist()
        model_config = ModelConfig(
            model_type="cnn",
            hidden_size=8,
            num_layers=1,
            kernel_size=3,
            dropout=0.0,
        )
        model = build_model(model_config)
        normalizer = TrajectoryNormalizer()
        thresholds = np.array([0.4, 0.6], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = os.path.join(temporary_directory, "model.pt")
            save_checkpoint(
                checkpoint,
                model,
                model_config=model_config,
                normalizer=normalizer,
                thresholds=thresholds,
                metadata={"time_column": "time_h"},
            )
            loaded_model, loaded_normalizer, loaded_thresholds, _ = load_checkpoint(
                checkpoint,
                device="cpu",
            )
            result = predict_dataframe(
                loaded_model,
                dataframe,
                normalizer=loaded_normalizer,
                thresholds=loaded_thresholds,
                device="cpu",
            )
        self.assertEqual(result["source_marker"].tolist(), original_markers)
        self.assertEqual(len(result), len(dataframe))
        for column in (
            "pred_division_probability",
            "pred_excluded_probability",
            "pred_division_point",
            "pred_excluded_point",
        ):
            self.assertIn(column, result)
        self.assertTrue(
            result[["pred_division_probability", "pred_excluded_probability"]]
            .notna()
            .all()
            .all()
        )


if __name__ == "__main__":
    unittest.main()
