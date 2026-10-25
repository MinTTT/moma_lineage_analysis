
"""Helpers for fitting chronological frames into the available gallery width."""

from moma_lineage_analysis.app.image_review_gallery import (
    paginate_review_frames,
    review_display_width,
)


def test_review_display_width_uses_saved_aspect_ratio():
    assert review_display_width((160, 30), 480) == 90
    assert review_display_width((336, 62), 480) == 89
    assert review_display_width(None, 480) == 88


def test_pagination_packs_as_many_frames_as_fit():
    # Eleven 90 px images plus ten 4 px gaps occupy 1030 px; a twelfth
    # would require 1124 px and therefore belongs on the next page.
    assert paginate_review_frames([90] * 12, 1080) == [tuple(range(11)), (11,)]
    assert paginate_review_frames([90] * 12, 500) == [
        (0, 1, 2, 3, 4),
        (5, 6, 7, 8, 9),
        (10, 11),
    ]


def test_pagination_handles_mixed_sizes_and_single_overflow():
    assert paginate_review_frames([100, 200, 50], 350) == [(0, 1), (2,)]
    # One image wider than the block remains alone and scrolls horizontally.
    assert paginate_review_frames([500, 90], 300) == [(0,), (1,)]
    assert paginate_review_frames([], 300) == []


def test_pagination_uses_six_frame_fallback_until_width_is_known():
    assert paginate_review_frames([90] * 14, None) == [
        tuple(range(6)),
        tuple(range(6, 12)),
        (12, 13),
    ]
