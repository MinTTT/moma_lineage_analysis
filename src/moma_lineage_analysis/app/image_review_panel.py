"""Independent Streamlit dialog for inspecting selected side-channel frames."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import streamlit as st

from moma_lineage_analysis.app.image_review_gallery import (
    image_data_url,
    paginate_review_frames,
    render_image_gallery,
    review_display_width,
)
from moma_lineage_analysis.image_review_dt import get_dt_meshes, render_dt_overlay
from moma_lineage_analysis import image_review_export as _export_module

# A running Streamlit process may still hold the earlier RGB-only helper even
# after this panel reloads. Refresh only that outdated module, not the session.
if not getattr(_export_module, "EXPORTS_ORIGINAL_IMAGE", False):
    import importlib

    _export_module = importlib.reload(_export_module)
export_review_pair = _export_module.export_review_pair
from moma_lineage_analysis.image_review import (
    ImageReviewError,
    ReviewSnapshot,
    get_review_frame,
    get_review_image_shape,
    load_review_pickle,
    optical_channels_for_frames,
    render_review_overlay,
)

PAGE_STATE_VERSION = 2


@st.cache_resource(max_entries=1, show_spinner=False, scope="session")
def _cached_export(path: str, modified_ns: int, size: int) -> dict:
    """Keep one FOV per session; file changes invalidate the cache key."""
    export = load_review_pickle(path)
    stat = Path(path).stat()
    if (stat.st_mtime_ns, stat.st_size) != (modified_ns, size):
        raise ImageReviewError("The image file changed while loading. Close and reopen image review.")
    return export


def close_image_review() -> None:
    # Dialog widget keys are cleaned up when it closes. Save choices separately.
    for name in ("fill", "outline", "dt", "cells", "height"):
        key = f"image_review_{name}"
        if key in st.session_state:
            st.session_state[f"image_review_preference_{name}"] = st.session_state[key]
    st.session_state.pop("image_review_snapshot", None)
    st.session_state.image_review_open = False
    st.session_state.pop("image_review_selected_frame", None)


def open_image_review(snapshot: ReviewSnapshot) -> None:
    """Capture a new review independently of the live plot selection."""
    st.session_state.image_review_snapshot = snapshot
    st.session_state.image_review_open = True
    st.session_state.image_review_page = 1
    st.session_state.pop("image_review_page_selector", None)
    st.session_state.image_review_selected_frame = None
    st.session_state.pop("image_review_gallery", None)
    st.session_state.pop("image_review_optical_channel", None)
    # Restore durable preferences after Streamlit cleans up closed dialog widgets.
    defaults = {"fill": False, "outline": True, "dt": False, "cells": "Selected cells", "height": 480}
    for name, default in defaults.items():
        st.session_state[f"image_review_{name}"] = st.session_state.get(
            f"image_review_preference_{name}", default
        )


def _page_number(value, page_count: int) -> int:
    """Recover old display labels and clamp stale selections after repagination."""
    if isinstance(value, str):
        label = re.fullmatch(r"Page +([0-9]+)(?: +of +[0-9]+)?", value.strip())
        if label:
            value = label.group(1)
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        number = 1
    return min(page_count, max(1, number))


def _select_page() -> None:
    st.session_state.image_review_page = _page_number(
        st.session_state.get("image_review_page_selector", 1),
        st.session_state.get("image_review_page_count", 1),
    )


def _change_page(step: int) -> None:
    if st.session_state.get("image_review_snapshot") is None:
        return
    pages = max(1, int(st.session_state.get("image_review_page_count", 1)))
    current = _page_number(st.session_state.get("image_review_page", 1), pages)
    st.session_state.image_review_page = _page_number(current + step, pages)


def _frame_info(snapshot: ReviewSnapshot, frame, optical: str, messages=()) -> str:
    labels = ", ".join(str(observation.cell_index) for observation in frame.observations)
    time = f"{frame.time_h:.3f} h" if frame.time_h is not None else "Time unavailable"
    text = (
        f"{Path(snapshot.source_csv).name.removesuffix('_cells_df.csv')} · "
        f"Image: {frame.time_point} · Time: {time} · "
        f"Side channel: {frame.side_channel} · Optical channel: {optical} · Cell: {labels}"
    )
    excluded = [str(item.cell_index) for item in frame.observations if item.excluded]
    divisions = [str(item.cell_index) for item in frame.observations if item.division]
    if excluded:
        text += f" · Excluded: {', '.join(excluded)}"
    if divisions:
        text += f" · Division: {', '.join(divisions)}"
    if messages:
        text += " · " + " · ".join(messages)
    return text


def _render_review(snapshot: ReviewSnapshot) -> None:
    st.caption(
        f"{snapshot.trajectory_key} · {len(snapshot.frames)} selected frames · "
        "earliest to latest, left to right"
    )
    st.caption("Snapshot of the selection when opened. Close and reopen to review new points or labels.")
    if st.button("Close image review", key="close_image_review"):
        close_image_review()
        st.rerun(scope="app")

    path = Path(snapshot.pickle_path)
    try:
        stat = path.stat()
        with st.spinner("Loading side-channel images…"):
            export = _cached_export(str(path), stat.st_mtime_ns, stat.st_size)
        channels = optical_channels_for_frames(export, snapshot.frames)
    except (OSError, ImageReviewError) as exc:
        st.warning(f"Image review is unavailable: {exc}")
        st.caption(f"Expected image file: {path.name}. Keep it beside the original cell measurements CSV.")
        return

    if st.session_state.get("image_review_optical_channel") not in channels:
        st.session_state.image_review_optical_channel = channels[0]
    channel_col, cells_col = st.columns([1, 1])
    with channel_col:
        optical = st.selectbox(
            "Optical channel", options=channels, key="image_review_optical_channel"
        )
    with cells_col:
        cells = st.selectbox(
            "Cells shown", options=("Selected cells", "All cells"), key="image_review_cells"
        )
    fill_col, outline_col, dt_col = st.columns([1, 1, 2])
    with fill_col:
        fill = st.toggle("Mask fill", key="image_review_fill")
    with outline_col:
        outline = st.toggle("Mask outlines", key="image_review_outline")
    with dt_col:
        dt = st.toggle(
            "DT spine and ribs", key="image_review_dt",
            help="Show the saved DT mesh ribs (yellow) and spine (magenta).",
        )

    height = st.select_slider(
        "Image height (px)", options=(360, 480, 600, 720), key="image_review_height"
    )
    # Compute display widths from the saved aspect ratios, then let the gallery's
    # measured width determine how many chronological frames fit on each page.
    frame_widths = []
    for frame in snapshot.frames:
        try:
            shape = get_review_image_shape(export, frame, optical)
        except ImageReviewError:
            shape = None
        frame_widths.append(review_display_width(shape, height))
    available_width = st.session_state.get("image_review_viewport_width")
    pages = paginate_review_frames(frame_widths, available_width) or [()]
    page_count = len(pages)
    st.session_state.image_review_page_count = page_count
    page = _page_number(st.session_state.get("image_review_page", 1), page_count)
    st.session_state.image_review_page = page
    page_indices = pages[page - 1]
    page_frames = [snapshot.frames[index] for index in page_indices]

    earlier_col, page_col, later_col = st.columns([1, 2, 1])
    with earlier_col:
        st.button(
            "◀ Earlier frames", key="image_review_earlier", disabled=page <= 1,
            on_click=_change_page, args=(-1,), width="stretch",
        )
    with page_col:
        if page_count > 1:
            # Keep widget labels separate from the canonical numeric page. A
            # changing total in format_func can leave stale browser labels that
            # Streamlit deserializes as strings instead of option values.
            st.session_state.image_review_page_selector = page
            st.selectbox(
                "Frame page", options=range(1, page_count + 1),
                key="image_review_page_selector", label_visibility="collapsed",
                format_func=lambda number: f"Page {number}", on_change=_select_page,
            )
        st.caption(f"Page {page} of {page_count}")
    with later_col:
        st.button(
            "Later frames ▶", key="image_review_later", disabled=page >= page_count,
            on_click=_change_page, args=(1,), width="stretch",
        )
    first_frame = page_indices[0] + 1 if page_indices else 0
    last_frame = page_indices[-1] + 1 if page_indices else 0
    st.caption(
        f"Frames {first_frame}–{last_frame} of {len(snapshot.frames)} · "
        f"{len(page_frames)} images on this page. Click an image for details. "
        "Selected cells: cyan · other cells: orange."
    )

    # Use the same contrast limits across the page for comparable intensities.
    loaded = []
    samples = []
    for frame in page_frames:
        try:
            image, mask, warnings = get_review_frame(export, frame, optical)
            loaded.append((image, mask, warnings, None))
            sample = np.asarray(image, dtype=float).ravel()[::max(1, image.size // 20000)]
            samples.append(sample[np.isfinite(sample)])
        except ImageReviewError as exc:
            loaded.append((None, None, (), str(exc)))
    finite_samples = [sample for sample in samples if sample.size]
    contrast = None
    if finite_samples:
        limits = np.percentile(np.concatenate(finite_samples), [1, 99])
        if limits[1] > limits[0]:
            contrast = (float(limits[0]), float(limits[1]))
    missing_count = sum(item[3] is not None for item in loaded)
    if missing_count:
        st.warning(f"{missing_count} of {len(page_frames)} frames on this page lack a usable {optical} image.")
    st.caption("Display contrast uses the 1st–99th percentiles across the images on this page.")

    gallery_frames = []
    source_frames = {}
    mask_frames = {}
    diagnostics = []
    for frame_index, frame, item in zip(page_indices, page_frames, loaded):
        image, mask, messages, error = item
        mask_frames[str(frame_index)] = mask
        messages = list(messages)
        source = None
        display_width = 88
        if error is not None:
            messages.append(error)
        else:
            source_frames[str(frame_index)] = image
            labels = tuple(observation.cell_index for observation in frame.observations)
            rgb = render_review_overlay(
                image, mask, labels, fill=fill, outline=outline,
                all_cells=cells == "All cells", contrast_limits=contrast,
            )
            if dt:
                meshes, dt_messages = get_dt_meshes(export, frame, all_cells=cells == "All cells")
                messages.extend(dt_messages)
                rgb = render_dt_overlay(rgb, meshes)
            display_width = review_display_width(rgb.shape[:2], height)
            source = image_data_url(rgb)
        diagnostics.extend(f"{frame.time_point}: {message}" for message in messages)
        gallery_frames.append({
            "id": str(frame_index),
            "src": source,
            "width": display_width,
            "height": height,
            "alt": f"Side channel {frame.side_channel}, frame {frame_index + 1}",
            "info": _frame_info(snapshot, frame, optical, messages),
        })
    if dt:
        st.caption("DT overlay: spine magenta · ribs yellow.")
    render_image_gallery(gallery_frames, reported_width=available_width)
    selected_id = st.session_state.get("image_review_selected_frame")
    has_image = isinstance(selected_id, str) and selected_id in source_frames
    has_mask = has_image and mask_frames.get(selected_id) is not None
    can_export = has_image and has_mask
    export_col, note_col = st.columns([1, 2])
    with export_col:
        if st.button(
            "Export selected image + mask (TIFF)",
            key="export_review_image",
            disabled=not can_export,
            help="Save original image pixels and the original label mask, without overlays, into the source data folder.",
        ):
            frame_index = int(selected_id)
            frame = snapshot.frames[frame_index]
            try:
                image_output, mask_output = export_review_pair(
                    source_frames[selected_id],
                    mask_frames[selected_id],
                    output_dir=Path(snapshot.source_csv).parent,
                    fov_name=Path(snapshot.source_csv).name.removesuffix("_cells_df.csv"),
                    time_point=frame.time_point,
                    optical_channel=optical,
                    side_channel=frame.side_channel,
                )
            except (OSError, ValueError) as exc:
                st.warning(f"TIFF export failed: {exc}")
            else:
                st.success(f"Saved {image_output.name} and {mask_output.name} to {image_output.parent}")
    with note_col:
        if not has_image:
            note = "Select an image on this page to enable TIFF export."
        elif not has_mask:
            note = "No mask is available for this image; TIFF pair export is disabled."
        else:
            note = "Original image only—no overlays. Image and mask share an 8-character tag; mask ends in _masks."
        st.caption(note)
    if diagnostics:
        with st.expander(f"Image notes ({len(diagnostics)})", expanded=False):
            for message in dict.fromkeys(diagnostics):
                st.warning(message)


@st.dialog("Selected cell image review", width="large", on_dismiss=close_image_review)
def show_image_review_dialog() -> None:
    snapshot = st.session_state.get("image_review_snapshot")
    if snapshot is not None:
        _render_review(snapshot)


def render_image_review_if_open(current_trajectory_key: str) -> None:
    if st.session_state.get("image_review_open", False):
        snapshot = st.session_state.get("image_review_snapshot")
        if snapshot is None or snapshot.trajectory_key != current_trajectory_key:
            close_image_review()
            return
        show_image_review_dialog()
