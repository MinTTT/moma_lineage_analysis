"""Focused tests for the side-channel image review backend."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from moma_lineage_analysis.image_review import (
    ImageReviewError,
    ReviewFrame,
    ReviewObservation,
    get_review_frame,
    load_review_pickle,
    make_review_snapshot,
    normalize_review_image,
    optical_channels_for_frames,
    render_review_overlay,
)


def make_csv(directory: Path, name: str = "FOV_007_cells_df.csv") -> Path:
    path = directory / name
    path.write_text("time_s,time_point,channels,cell_index\n", encoding="utf-8")
    return path


def make_trajectory(rows: list[dict]) -> dict:
    return {
        "trajectory": pd.DataFrame(rows),
        "excluded_indices": {4},
        "div_indices": [10],
    }


class SnapshotTests(unittest.TestCase):
    def test_snapshot_dedupes_groups_annotations_and_sorts_naturally(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source = make_csv(Path(raw))
            rows = [
                # t10 appears before t2 in insertion order; sorting must still
                # place t2 first rather than using lexical frame names.
                {"fov_cell_index": 10, "channels": "003", "time_point": "t10", "cell_index": 2, "time_h": 10 / 60, "time_s": 600.0},
                {"fov_cell_index": 2, "channels": "003", "time_point": "t2", "cell_index": 1, "time_h": 2 / 60, "time_s": 120.0},
                {"fov_cell_index": 3, "channels": "003", "time_point": "t2", "cell_index": 7, "time_h": 2 / 60, "time_s": 120.0},
                {"fov_cell_index": 4, "channels": "003", "time_point": "t4", "cell_index": 8, "time_h": 4 / 60, "time_s": 240.0},
            ]
            snapshot = make_review_snapshot(
                "FOV_007_CH_003",
                make_trajectory(rows),
                [10, 2, 3, 3, 4, 10],
                source,
            )
        self.assertEqual(snapshot.pickle_path, str(source.with_name("FOV_007_exported_data.pkl")))
        self.assertEqual([frame.time_point for frame in snapshot.frames], ["t2", "t4", "t10"])
        self.assertEqual(len(snapshot.frames[0].observations), 2)
        self.assertEqual(snapshot.frames[0].observations[1].cell_index, 7)
        self.assertTrue(snapshot.frames[1].observations[0].excluded)
        self.assertFalse(snapshot.frames[1].observations[0].division)
        self.assertTrue(snapshot.frames[2].observations[0].division)
        self.assertIsInstance(snapshot.frames, tuple)
        self.assertIsInstance(snapshot.frames[0].observations, tuple)

    def test_snapshot_preserves_padded_source_stem_and_requires_csv(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            source = make_csv(directory, "FOV_0009_cells_df.csv")
            row = {"fov_cell_index": 1, "channels": 3, "time_point": "t1", "cell_index": 1, "time_h": 0.0, "time_s": 0.0}
            snapshot = make_review_snapshot("key", make_trajectory([row]), [1], source)
            self.assertTrue(snapshot.pickle_path.endswith("FOV_0009_exported_data.pkl"))
            wrong_suffix = directory / "FOV_0009.csv"
            wrong_suffix.write_text("x\n", encoding="utf-8")
            with self.assertRaisesRegex(ImageReviewError, "_cells_df.csv"):
                make_review_snapshot("key", make_trajectory([row]), [1], wrong_suffix)
            with self.assertRaisesRegex(ImageReviewError, "Process Data"):
                make_review_snapshot("key", make_trajectory([row]), [1], None)
            with self.assertRaisesRegex(ImageReviewError, "Cannot locate"):
                make_review_snapshot(
                    "key", make_trajectory([row]), [1], directory / "missing_cells_df.csv"
                )

    def test_snapshot_rejects_ambiguous_duplicate_and_missing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source = make_csv(Path(raw))
            row = {"fov_cell_index": 1, "channels": 3, "time_point": "t1", "cell_index": 1, "time_h": 0.0, "time_s": 0.0}
            with self.assertRaisesRegex(ImageReviewError, "ambiguous duplicate"):
                make_review_snapshot("key", make_trajectory([row, dict(row)]), [1], source)
            bad_label = dict(row, cell_index=0)
            with self.assertRaisesRegex(ImageReviewError, "positive"):
                make_review_snapshot("key", make_trajectory([bad_label]), [1], source)
            bad_time = dict(row, time_point=np.nan)
            with self.assertRaises(ImageReviewError):
                make_review_snapshot("key", make_trajectory([bad_time]), [1], source)
            subset = make_trajectory([row])
            subset["trajectory"] = subset["trajectory"].drop(columns=["cell_index"])
            with self.assertRaisesRegex(ImageReviewError, "cell_index"):
                make_review_snapshot("key", subset, [1], source)

    def test_snapshot_uses_consistent_seconds_when_hours_missing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source = make_csv(Path(raw))
            rows = [
                {"fov_cell_index": 10, "channels": "003", "time_point": "t10", "cell_index": 2, "time_h": np.nan, "time_s": 600.0},
                {"fov_cell_index": 2, "channels": "003", "time_point": "t2", "cell_index": 1, "time_h": np.nan, "time_s": 120.0},
            ]
            snapshot = make_review_snapshot("key", make_trajectory(rows), [10, 2], source)
        self.assertEqual([frame.time_point for frame in snapshot.frames], ["t2", "t10"])

    def test_snapshot_orders_nd2_style_stems_when_times_missing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source = make_csv(Path(raw))
            rows = [
                {"fov_cell_index": 10, "channels": "003", "time_point": "t10.tiff", "cell_index": 2, "time_h": np.nan, "time_s": np.nan},
                {"fov_cell_index": 2, "channels": "003", "time_point": "t2.tiff", "cell_index": 1, "time_h": np.nan, "time_s": np.nan},
                {"fov_cell_index": 1, "channels": "003", "time_point": "t00001.tif", "cell_index": 3, "time_h": np.nan, "time_s": np.nan},
            ]
            snapshot = make_review_snapshot("key", make_trajectory(rows), [10, 2, 1], source)
        self.assertEqual(
            [frame.time_point for frame in snapshot.frames],
            ["t00001.tif", "t2.tiff", "t10.tiff"],
        )

    def test_snapshot_resolves_row_with_duplicate_dataframe_index_labels(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source = make_csv(Path(raw))
            rows = [
                {"fov_cell_index": 1, "channels": 3, "time_point": "t1", "cell_index": 11, "time_h": 0.0, "time_s": 0.0},
                {"fov_cell_index": 2, "channels": 3, "time_point": "t2", "cell_index": 22, "time_h": 1 / 60, "time_s": 60.0},
            ]
            data = make_trajectory(rows)
            data["trajectory"].index = [7, 7]
            snapshot = make_review_snapshot("key", data, [2], source)
        self.assertEqual(snapshot.frames[0].time_point, "t2")
        self.assertEqual(snapshot.frames[0].observations[0].cell_index, 22)

    def test_snapshot_is_detached_from_mutable_dataframe_and_sets(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source = make_csv(Path(raw))
            row = {"fov_cell_index": 1, "channels": 3, "time_point": "t1", "cell_index": 1, "time_h": 0.0, "time_s": 0.0}
            data = make_trajectory([row])
            snapshot = make_review_snapshot("key", data, [1], source)
            data["trajectory"].loc[0, "cell_index"] = 99
            data["excluded_indices"].add(1)
            data["div_indices"].append(1)
        self.assertEqual(snapshot.frames[0].observations[0].cell_index, 1)
        self.assertFalse(snapshot.frames[0].observations[0].excluded)
        self.assertFalse(snapshot.frames[0].observations[0].division)


def make_export() -> dict:
    image = np.arange(16, dtype=np.uint16).reshape(4, 4)
    mask = np.zeros((4, 4), dtype=np.int32)
    mask[:2, :2] = 1
    mask[:2, 2:] = 2
    mask[2:, :] = 3
    return {
        "info_dict": {
            "cell_detection_opticalChannel_name": "bf",
            "opticalChannels": ["bf", "gfp"],
        },
        "cells_sideChannel_image_dict": {
            "fov_channels_image_dict": {
                3: {
                    "t2": {"bf": image, "gfp": image + 100},
                    "t10": {"bf": image + 200},
                },
                "side_B": {"t1": {"bf": image + 300}},
            },
            "fov_cells_mask_dict": {
                3: {"t2": mask.copy(), "t10": mask.copy()},
                "side_B": {"t1": mask.copy()},
            },
        },
    }


def review_frame(
    side=3,
    time_point="t2",
    labels=(1, 2),
    time_h=1 / 30,
    time_s=120.0,
) -> ReviewFrame:
    observations = tuple(
        ReviewObservation(
            fov_cell_index=100 + index,
            cell_index=label,
            time_h=time_h,
            time_s=time_s,
            excluded=False,
            division=False,
        )
        for index, label in enumerate(labels)
    )
    return ReviewFrame(
        side_channel=side,
        time_point=time_point,
        time_h=time_h,
        time_s=time_s,
        observations=observations,
    )


class PickleAndChannelTests(unittest.TestCase):
    def test_bundled_pickle_smoke_and_shape_validation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "FOV_1_exported_data.pkl"
            with path.open("wb") as handle:
                pickle.dump(make_export(), handle)
            loaded = load_review_pickle(path)
            self.assertEqual(loaded["info_dict"]["cell_detection_opticalChannel_name"], "bf")
            bad = Path(raw) / "bad_exported_data.pkl"
            bad.write_bytes(b"not a pickle")
            with self.assertRaisesRegex(ImageReviewError, "Could not read"):
                load_review_pickle(bad)
            shape = Path(raw) / "shape_exported_data.pkl"
            with shape.open("wb") as handle:
                pickle.dump({"info_dict": {}, "cells_sideChannel_image_dict": {}}, handle)
            with self.assertRaisesRegex(ImageReviewError, "images mapping"):
                load_review_pickle(shape)

    def test_optical_channel_union_detection_first_and_missing_frame_tolerated(self) -> None:
        frames = [
            review_frame(side="3", time_point="t10"),
            review_frame(side="003", time_point="missing"),
            review_frame(side=3, time_point="t2"),
        ]
        channels = optical_channels_for_frames(make_export(), frames)
        self.assertEqual(channels, ("bf", "gfp"))
        export = make_export()
        export["cells_sideChannel_image_dict"]["fov_channels_image_dict"] = {}
        with self.assertRaisesRegex(ImageReviewError, "None of the selected frames"):
            optical_channels_for_frames(export, frames)

    def test_detection_channel_is_not_invented_when_unavailable(self) -> None:
        export = make_export()
        export["info_dict"]["cell_detection_opticalChannel_name"] = "missing"
        channels = optical_channels_for_frames(export, [review_frame()])
        self.assertEqual(channels, ("bf", "gfp"))

    def test_optical_union_rejects_ambiguous_image_aliases(self) -> None:
        export = make_export()
        export["cells_sideChannel_image_dict"]["fov_channels_image_dict"]["003"] = {}
        with self.assertRaisesRegex(ImageReviewError, "duplicate numeric aliases"):
            optical_channels_for_frames(export, [review_frame()])

    def test_side_key_numeric_conversion_and_alias_rejection(self) -> None:
        image, _, _ = get_review_frame(make_export(), review_frame(side="003"), "bf")
        self.assertEqual(image.dtype, np.uint16)
        export = make_export()
        export["cells_sideChannel_image_dict"]["fov_cells_mask_dict"]["003"] = {}
        export["cells_sideChannel_image_dict"]["fov_cells_mask_dict"]["3"] = {}
        with self.assertRaisesRegex(ImageReviewError, "duplicate numeric aliases"):
            get_review_frame(export, review_frame(side=3), "bf")

    def test_mixed_image_and_mask_keys_resolve_independently(self) -> None:
        export = make_export()
        masks = export["cells_sideChannel_image_dict"]["fov_cells_mask_dict"]
        masks["003"] = masks.pop(3)
        image, mask, messages = get_review_frame(export, review_frame(side=3), "bf")
        self.assertIsNotNone(image)
        self.assertIsNotNone(mask)
        self.assertEqual(messages, ())

        ambiguous_images = make_export()
        ambiguous_images["cells_sideChannel_image_dict"]["fov_channels_image_dict"]["003"] = {}
        with self.assertRaisesRegex(
            ImageReviewError, "duplicate numeric aliases"
        ):
            get_review_frame(ambiguous_images, review_frame(side=3), "bf")

    def test_image_only_side_with_missing_mask_still_renders_image(self) -> None:
        export = make_export()
        del export["cells_sideChannel_image_dict"]["fov_cells_mask_dict"][3]
        image, mask, messages = get_review_frame(export, review_frame(side="003"), "bf")
        self.assertIsNotNone(image)
        self.assertIsNone(mask)
        self.assertTrue(any("Mask missing" in message for message in messages))

        empty_masks = make_export()
        empty_masks["cells_sideChannel_image_dict"]["fov_cells_mask_dict"] = {}
        image, mask, messages = get_review_frame(empty_masks, review_frame(side=3), "bf")
        self.assertIsNotNone(image)
        self.assertIsNone(mask)
        self.assertTrue(any("Mask missing" in message for message in messages))

    def test_exact_time_and_optical_names_never_guessed(self) -> None:
        with self.assertRaisesRegex(ImageReviewError, "time_point matching is exact"):
            get_review_frame(make_export(), review_frame(time_point="t02"), "bf")
        with self.assertRaisesRegex(ImageReviewError, "Optical channel"):
            get_review_frame(make_export(), review_frame(), "red")


class FrameRetrievalTests(unittest.TestCase):
    def test_returns_identical_image_and_mask_alignment(self) -> None:
        export = make_export()
        image, mask, messages = get_review_frame(export, review_frame(labels=(1, 99)), "bf")
        self.assertIs(image, export["cells_sideChannel_image_dict"]["fov_channels_image_dict"][3]["t2"]["bf"])
        self.assertIs(mask, export["cells_sideChannel_image_dict"]["fov_cells_mask_dict"][3]["t2"])
        self.assertEqual(image.shape, mask.shape)
        self.assertEqual(len(messages), 1)
        self.assertIn("99", messages[0])

    def test_missing_or_mismatched_mask_returns_diagnostics_with_image(self) -> None:
        export = make_export()
        export["cells_sideChannel_image_dict"]["fov_cells_mask_dict"][3]["t2"] = np.zeros((2, 2))
        image, mask, messages = get_review_frame(export, review_frame(), "bf")
        self.assertIsNotNone(image)
        self.assertIsNone(mask)
        self.assertTrue(any("Mask malformed" in message for message in messages))

        export = make_export()
        del export["cells_sideChannel_image_dict"]["fov_cells_mask_dict"][3]["t2"]
        image, mask, messages = get_review_frame(export, review_frame(), "bf")
        self.assertIsNotNone(image)
        self.assertIsNone(mask)
        self.assertTrue(any("Mask missing" in message for message in messages))

    def test_invalid_mask_types_return_diagnostics_not_renderer_crashes(self) -> None:
        invalid_masks = [
            np.zeros((4, 4), dtype=bool),
            np.full((4, 4), "1", dtype=object),
            np.full((4, 4), "1", dtype=str),
            np.full((4, 4), -1, dtype=np.int32),
            np.full((4, 4), 1 + 0j, dtype=np.complex128),
            np.full((4, 4), np.nan, dtype=np.float64),
            np.full((4, 4), 1.5, dtype=np.float64),
            np.zeros((0, 0), dtype=np.int32),
        ]
        for invalid in invalid_masks:
            with self.subTest(dtype=getattr(invalid, "dtype", None)):
                export = make_export()
                export["cells_sideChannel_image_dict"]["fov_cells_mask_dict"][3]["t2"] = invalid
                image, mask, messages = get_review_frame(export, review_frame(), "bf")
                self.assertIsNotNone(image)
                self.assertIsNone(mask)
                self.assertTrue(any("Mask malformed" in message for message in messages))

        for invalid_image in [
            np.ones((4, 4), dtype=np.complex128),
            np.zeros((0, 0), dtype=np.uint16),
        ]:
            with self.subTest(image_dtype=getattr(invalid_image, "dtype", None)):
                export = make_export()
                export["cells_sideChannel_image_dict"]["fov_channels_image_dict"][3]["t2"]["bf"] = invalid_image
                with self.assertRaisesRegex(ImageReviewError, "real numeric|must not be empty"):
                    get_review_frame(export, review_frame(), "bf")

    def test_missing_image_raises_for_per_frame_ui_warning(self) -> None:
        export = make_export()
        del export["cells_sideChannel_image_dict"]["fov_channels_image_dict"][3]["t2"]
        with self.assertRaisesRegex(ImageReviewError, "time_point matching is exact"):
            get_review_frame(export, review_frame(), "bf")


class NormalizationAndOverlayTests(unittest.TestCase):
    def test_uint16_constant_nonfinite_and_input_not_mutated(self) -> None:
        image = np.array([[0.0, np.nan], [1000.0, np.inf]], dtype=np.float32)
        before = image.copy()
        normalized = normalize_review_image(image)
        self.assertEqual(normalized.dtype, np.uint8)
        np.testing.assert_array_equal(image, before)
        self.assertEqual(normalized[0, 0], 0)
        self.assertEqual(normalized[1, 0], 255)
        self.assertEqual(normalized[0, 1], 0)

        wide = np.array([[0, 65535], [32768, 100]], dtype=np.uint16)
        normalized_wide = normalize_review_image(wide)
        self.assertEqual(normalized_wide[0, 0], 0)
        self.assertEqual(normalized_wide[0, 1], 255)
        constant = np.full((3, 3), 42, dtype=np.uint16)
        self.assertTrue(np.all(normalize_review_image(constant) == 42))
        nonfinite = np.array([[np.nan, np.inf], [-np.inf, np.nan]])
        self.assertTrue(np.all(normalize_review_image(nonfinite) == 127))

    def test_shared_limits_are_respected(self) -> None:
        low = np.array([[0.0, 50.0]])
        high = np.array([[50.0, 100.0]])
        a = normalize_review_image(low, (0.0, 100.0))
        b = normalize_review_image(high, (0.0, 100.0))
        self.assertEqual(a[0, 0], 0)
        self.assertEqual(a[0, 1], 128)
        self.assertEqual(b[0, 1], 255)
        with self.assertRaisesRegex(ImageReviewError, "finite"):
            normalize_review_image(low, (0.0, np.nan))

    def test_touching_selected_labels_keep_distinct_outlines(self) -> None:
        image = np.zeros((5, 6), dtype=np.uint8)
        mask = np.zeros((5, 6), dtype=np.int32)
        mask[1:4, 1:3] = 1
        mask[1:4, 3:5] = 2
        rendered = render_review_overlay(image, mask, [1, 2])
        cyan = np.array([0, 220, 255], dtype=np.uint8)
        # The shared border is preserved as a cyan label-neighbor boundary.
        np.testing.assert_array_equal(rendered[2, 2], cyan)
        np.testing.assert_array_equal(rendered[2, 3], cyan)
        self.assertGreater(np.sum(np.all(rendered == cyan, axis=2)), 4)
        self.assertEqual(rendered.dtype, np.uint8)
        self.assertEqual(rendered.shape, (5, 6, 3))
        self.assertEqual(image.max(), 0)

    def test_all_cells_outline_and_fill_do_not_mutate_sources(self) -> None:
        image = np.arange(36, dtype=np.uint8).reshape(6, 6)
        mask = np.zeros((6, 6), dtype=np.int32)
        mask[1:3, 1:3] = 1
        mask[3:5, 3:5] = 2
        image_before = image.copy()
        mask_before = mask.copy()
        rendered = render_review_overlay(
            image,
            mask,
            [1],
            fill=True,
            outline=True,
            all_cells=True,
            opacity=0.5,
        )
        np.testing.assert_array_equal(image, image_before)
        np.testing.assert_array_equal(mask, mask_before)
        np.testing.assert_array_equal(rendered[1, 1], np.array([0, 220, 255], dtype=np.uint8))
        self.assertTrue(np.any(np.all(rendered == np.array([255, 165, 0], dtype=np.uint8), axis=2)))
        self.assertGreater(rendered[2, 2, 2], 0)  # selected fill contributes blue channel

    def test_all_cells_fill_colors_neighbors_with_outlines_off(self) -> None:
        image = np.zeros((4, 4), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0:2, 0:2] = 1
        mask[2:4, 2:4] = 2
        rendered = render_review_overlay(
            image, mask, [1], fill=True, outline=False, all_cells=True, opacity=1.0
        )
        np.testing.assert_array_equal(rendered[0, 0], [0, 220, 255])
        np.testing.assert_array_equal(rendered[3, 3], [255, 165, 0])
        np.testing.assert_array_equal(rendered[0, 3], [0, 0, 0])

    def test_image_edge_boundary_is_outlined(self) -> None:
        image = np.zeros((3, 3), dtype=np.uint8)
        mask = np.zeros((3, 3), dtype=np.int32)
        mask[0, :] = 1
        rendered = render_review_overlay(image, mask, [1])
        cyan = np.array([0, 220, 255], dtype=np.uint8)
        np.testing.assert_array_equal(rendered[0, 0], cyan)

    def test_high_integer_labels_have_exact_boundaries_without_interior_fill(self) -> None:
        high = 2**62
        image = np.zeros((3, 3), dtype=np.uint8)
        mask = np.full((3, 3), high, dtype=np.uint64)
        mask[1, 1] = high + 1
        rendered = render_review_overlay(image, mask, [high])
        cyan = np.array([0, 220, 255], dtype=np.uint8)
        black = np.array([0, 0, 0], dtype=np.uint8)
        np.testing.assert_array_equal(rendered[0, 0], cyan)
        np.testing.assert_array_equal(rendered[1, 1], black)

    def test_overlay_rejects_shape_mismatch_and_bad_labels(self) -> None:
        image = np.zeros((3, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ImageReviewError, "matching image shape"):
            render_review_overlay(image, np.zeros((2, 2), dtype=np.int32), [1])
        with self.assertRaisesRegex(ImageReviewError, "positive"):
            render_review_overlay(image, None, [0])
        unchanged = render_review_overlay(image, None, [1], outline=True, fill=True)
        self.assertEqual(unchanged.shape, (3, 3, 3))
        self.assertTrue(np.all(unchanged == 0))

    def test_overlay_rejects_invalid_mask_and_image_arrays(self) -> None:
        image = np.zeros((3, 3), dtype=np.uint8)
        invalid_masks = [
            np.zeros((3, 3), dtype=bool),
            np.full((3, 3), "1", dtype=object),
            np.array([["1", "1", "1"], ["1", "1", "1"], ["1", "1", "1"]], dtype=str),
            np.full((3, 3), -1, dtype=np.int32),
            np.full((3, 3), 1 + 0j, dtype=np.complex128),
        ]
        for invalid in invalid_masks:
            with self.subTest(dtype=getattr(invalid, "dtype", None), shape=invalid.shape):
                with self.assertRaises(ImageReviewError):
                    render_review_overlay(image, invalid, [1])
        with self.assertRaisesRegex(ImageReviewError, "real numeric"):
            render_review_overlay(np.ones((3, 3), dtype=np.complex128), None, [1])
        with self.assertRaisesRegex(ImageReviewError, "empty"):
            normalize_review_image(np.zeros((0, 0), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
