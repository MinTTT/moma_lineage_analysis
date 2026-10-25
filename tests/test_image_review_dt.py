"""Focused tests for saved DT spine/rib geometry and its review overlay."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from moma_lineage_analysis.image_review import ImageReviewError, ReviewFrame, ReviewObservation
from moma_lineage_analysis.image_review_dt import (
    DTMesh,
    DT_RIB_COLOR,
    DT_SPINE_COLOR,
    get_dt_meshes,
    render_dt_overlay,
)


def observation(cell_index: int) -> ReviewObservation:
    return ReviewObservation(
        fov_cell_index=cell_index,
        cell_index=cell_index,
        time_h=None,
        time_s=None,
        excluded=False,
        division=False,
    )


def frame(*cell_indices: int, side_channel=3, time_point="t002") -> ReviewFrame:
    return ReviewFrame(
        side_channel=side_channel,
        time_point=time_point,
        time_h=None,
        time_s=None,
        observations=tuple(observation(index) for index in cell_indices),
    )


_DEFAULT_MESH = [
    [1.0, 1.0, 3.0, 1.0],
    [2.0, 2.0, 4.0, 2.0],
    [3.0, 3.0, 5.0, 3.0],
]


def record(cell_index: int, mesh=_DEFAULT_MESH, **extra):
    data = {"cell_index": cell_index, "mesh": mesh}
    data.update(extra)
    return data


def export_with(records, *, side="003", time_point="t002") -> dict:
    return {
        "info_dict": {"cell_detection_opticalChannel_name": "phase"},
        "cells_sideChannel_image_dict": {
            "fov_cells_pars_dict": {side: {time_point: records}}
        },
    }


class GetDTMeshesTests(unittest.TestCase):
    def test_label_join_padded_alias_order_and_no_mutation(self) -> None:
        payload = export_with([
            record(9, [[10.0, 4.0, 12.0, 4.0]]),
            record(2),
            record(7, [[6.0, 5.0, 8.0, 5.0]]),
        ])
        before = copy.deepcopy(payload)
        # Deliberately asks for label 7 before label 2, in reverse list order.
        meshes, warnings = get_dt_meshes(payload, frame(7, 2))
        self.assertEqual([mesh.cell_index for mesh in meshes], [7, 2])
        self.assertEqual(warnings, ())
        self.assertIsInstance(meshes, tuple)
        self.assertEqual(meshes[1].ribs.dtype, np.float64)
        self.assertFalse(meshes[1].ribs.flags.writeable)
        np.testing.assert_array_equal(meshes[1].ribs, np.asarray(record(2)["mesh"], dtype=np.float64))
        self.assertEqual(payload, before)
        self.assertEqual(payload["cells_sideChannel_image_dict"]["fov_cells_pars_dict"]["003"]["t002"][0]["mesh"][0][0], 10.0)

    def test_default_uses_snapshot_labels_all_cells_expands_scope(self) -> None:
        payload = export_with([record(4), record(2), record(7)])
        meshes, warnings = get_dt_meshes(payload, frame(2, 8))
        self.assertEqual([mesh.cell_index for mesh in meshes], [2])
        self.assertEqual(len(warnings), 1)
        self.assertIn("Cell label 8", warnings[0])

        all_meshes, all_warnings = get_dt_meshes(payload, frame(2, 8), all_cells=True)
        self.assertEqual([mesh.cell_index for mesh in all_meshes], [2, 4, 7])
        self.assertEqual(len(all_warnings), 1)
        self.assertIn("Cell label 8", all_warnings[0])

    def test_empty_frame_is_noop_unless_all_cells_requested(self) -> None:
        payload = export_with([record(2)])
        self.assertEqual(get_dt_meshes(payload, frame()), ((), ()))
        meshes, warnings = get_dt_meshes(payload, frame(), all_cells=True)
        self.assertEqual([mesh.cell_index for mesh in meshes], [2])
        self.assertEqual(warnings, ())

    def test_missing_payload_side_time_and_legacy_mesh_are_diagnostics(self) -> None:
        empty = get_dt_meshes({}, frame(2))
        self.assertEqual(empty[0], ())
        self.assertIn("cells_sideChannel_image_dict", empty[1][0])

        legacy = export_with([])
        legacy["cells_sideChannel_image_dict"].pop("fov_cells_pars_dict")
        meshes, warnings = get_dt_meshes(legacy, frame(2))
        self.assertEqual(meshes, ())
        self.assertIn("fov_cells_pars_dict", warnings[0])

        meshes, warnings = get_dt_meshes(export_with([]), frame(2, side_channel=3), )
        self.assertEqual(meshes, ())
        self.assertIn("no saved parameter record", warnings[0])

        meshes, warnings = get_dt_meshes(export_with([record(2)], time_point="t001"), frame(2, time_point="t002"))
        self.assertEqual(meshes, ())
        self.assertIn("time matching is exact", warnings[0])

        meshes, warnings = get_dt_meshes(export_with([record(2, mesh=None)]), frame(2))
        self.assertEqual(meshes, ())
        self.assertIn("legacy geometry is not reconstructed", warnings[0])

    def test_malformed_invalid_and_duplicate_records_never_guess(self) -> None:
        payload = export_with([
            "not-a-record",
            {"cell_index": "2", "mesh": []},
            record(2, mesh=[[0.0, 0.0, 1.0, 1.0]]),
            record(2, mesh=[[5.0, 5.0, 6.0, 6.0]]),
            record(3, mesh=[[0.0, 0.0, 1.0]]),
            record(4, mesh=[[0.0, 0.0, np.nan, 1.0]]),
            record(5),
        ])
        meshes, warnings = get_dt_meshes(payload, frame(2, 3, 4, 5, 6))
        self.assertEqual([mesh.cell_index for mesh in meshes], [5])
        text = "\n".join(warnings)
        self.assertIn("malformed: expected a mapping", text)
        self.assertIn("positive integer cell_index", text)
        self.assertIn("Cell label 2", text)
        self.assertIn("duplicate parameter records", text)
        self.assertIn("Cell label 3", text)
        self.assertIn("shape (N, 4)", text)
        self.assertIn("Cell label 4", text)
        self.assertIn("finite", text)
        self.assertIn("Cell label 6", text)
        self.assertIn("no saved parameter record", text)

    def test_algorithm_metadata_is_optional_for_old_exports(self) -> None:
        payload = export_with([record(2)])
        payload["cells_sideChannel_image_dict"]["fov_cells_pars_dict"]["003"]["t002"][0]["mesh"] = [[1.0, 1.0, 2.0, 1.0]]
        payload.pop("info_dict")
        meshes, warnings = get_dt_meshes(payload, frame(2))
        self.assertEqual(len(meshes), 1)
        self.assertEqual(warnings, ())

    def test_invalid_api_inputs_raise(self) -> None:
        with self.assertRaisesRegex(ImageReviewError, "ReviewFrame"):
            get_dt_meshes({}, object())
        with self.assertRaisesRegex(ImageReviewError, "non-empty exact"):
            get_dt_meshes({}, frame(2, time_point=""))
        with self.assertRaisesRegex(ImageReviewError, "boolean"):
            get_dt_meshes({}, frame(2), all_cells=1)
        with self.assertRaisesRegex(ImageReviewError, "positive integer"):
            get_dt_meshes({}, frame(0))


class RenderDTOverlayTests(unittest.TestCase):
    def test_exact_x_y_ribs_spine_copy_and_no_mutation(self) -> None:
        image = np.full((7, 9, 3), 11, dtype=np.uint8)
        before = image.copy()
        # Midpoints are (2, 2) and (4, 4): row=Y, column=X.
        ribs = np.array([[1.0, 2.0, 3.0, 2.0], [3.0, 4.0, 5.0, 4.0]])
        result = render_dt_overlay(image, [DTMesh(2, ribs)])
        self.assertIsNot(result, image)
        np.testing.assert_array_equal(image, before)
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result[2, 1], DT_RIB_COLOR)
        np.testing.assert_array_equal(result[2, 2], DT_SPINE_COLOR)
        np.testing.assert_array_equal(result[2, 3], DT_RIB_COLOR)
        np.testing.assert_array_equal(result[3, 3], DT_SPINE_COLOR)
        np.testing.assert_array_equal(result[4, 3], DT_RIB_COLOR)
        np.testing.assert_array_equal(result[4, 4], DT_SPINE_COLOR)
        np.testing.assert_array_equal(result[4, 5], DT_RIB_COLOR)
        # No accidental X/Y swap or background painting.
        np.testing.assert_array_equal(result[1, 2], 11)
        np.testing.assert_array_equal(result[0, 0], 11)

    def test_single_rib_cell_draws_rib_and_midpoint(self) -> None:
        result = render_dt_overlay(
            np.zeros((5, 7, 3), dtype=np.uint8),
            [DTMesh(1, np.array([[1.0, 2.0, 5.0, 2.0]])),],
        )
        np.testing.assert_array_equal(result[2, 1], DT_RIB_COLOR)
        np.testing.assert_array_equal(result[2, 3], DT_SPINE_COLOR)
        np.testing.assert_array_equal(result[2, 5], DT_RIB_COLOR)

    def test_two_cells_are_not_connected(self) -> None:
        first = np.array([[0.0, 1.0, 2.0, 1.0]])
        second = np.array([[6.0, 4.0, 8.0, 4.0]])
        result = render_dt_overlay(np.zeros((6, 10, 3), dtype=np.uint8), [DTMesh(1, first), DTMesh(2, second)])
        self.assertFalse((result == DT_SPINE_COLOR).all(axis=2)[1:5, 3:5].any())

    def test_absurd_coordinates_clip_before_rasterizing(self) -> None:
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        ribs = np.array([[-1e300, 0.0, 1e300, 0.0], [-100.0, 2.0, 2.0, 2.0]])
        result = render_dt_overlay(image, [DTMesh(2, ribs)])
        # The huge rib clips to row 0 and its midpoint paints only the clipped
        # start; the other rib clips from x=-100 to the in-bounds x=2 endpoint.
        np.testing.assert_array_equal(result[0, 0], DT_SPINE_COLOR)
        for column in range(1, 6):
            np.testing.assert_array_equal(result[0, column], DT_RIB_COLOR)
        self.assertTrue((result[2] == DT_RIB_COLOR).all(axis=1)[0:3].all())
        np.testing.assert_array_equal(result[2, 3], 0)
        np.testing.assert_array_equal(result[3, 3], 0)

    def test_invalid_renderer_inputs_raise(self) -> None:
        with self.assertRaisesRegex(ImageReviewError, "RGB uint8"):
            render_dt_overlay(np.zeros((4, 5), dtype=np.uint8), [])
        with self.assertRaisesRegex(ImageReviewError, "sequence"):
            render_dt_overlay(np.zeros((4, 5, 3), dtype=np.uint8), np.empty((1,)))
        with self.assertRaisesRegex(ImageReviewError, "DTMesh"):
            render_dt_overlay(np.zeros((4, 5, 3), dtype=np.uint8), [(2, np.zeros((1, 4)))])
        with self.assertRaisesRegex(ImageReviewError, "Duplicate cell label"):
            render_dt_overlay(
                np.zeros((4, 5, 3), dtype=np.uint8),
                [DTMesh(2, np.zeros((1, 4))), DTMesh(2, np.zeros((1, 4)))],
            )
        with self.assertRaisesRegex(ImageReviewError, r"shape \(N, 4\)"):
            render_dt_overlay(np.zeros((4, 5, 3), dtype=np.uint8), [DTMesh(2, np.zeros((0, 4)))])


if __name__ == "__main__":
    unittest.main()
