"""Compact, keyboard-accessible image strip with click-to-inspect metadata."""

from __future__ import annotations

import base64
from io import BytesIO
from weakref import WeakKeyDictionary

import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v2 as components


GALLERY_GAP_PX = 4
INITIAL_FRAMES_PER_PAGE = 6
MISSING_IMAGE_WIDTH_PX = 88


def review_display_width(image_shape, image_height: int) -> int:
    """Return the displayed width for one saved image, preserving aspect ratio."""

    if image_shape is None:
        return MISSING_IMAGE_WIDTH_PX
    rows, columns = image_shape
    if rows <= 0 or columns <= 0:
        raise ValueError(f"Image shape must be positive; got {image_shape!r}.")
    return max(1, round(image_height * columns / rows))


def paginate_review_frames(
    frame_widths, available_width: int | None, *, gap_px: int = GALLERY_GAP_PX
) -> list[tuple[int, ...]]:
    """Greedily pack consecutive frame indices into pages that fit the block.

    Widths are measured in display pixels. A single frame wider than the block
    gets its own horizontally scrollable page. Before the browser reports its
    width, small six-frame pages avoid an oversized initial payload.
    """

    widths = [max(1, int(width)) for width in frame_widths]
    if not widths:
        return []
    if available_width is None or available_width <= 0:
        return [
            tuple(range(start, min(start + INITIAL_FRAMES_PER_PAGE, len(widths))))
            for start in range(0, len(widths), INITIAL_FRAMES_PER_PAGE)
        ]
    pages: list[tuple[int, ...]] = []
    current: list[int] = []
    used_width = 0
    for index, width in enumerate(widths):
        added_width = width + (gap_px if current else 0)
        if current and used_width + added_width > available_width:
            pages.append(tuple(current))
            current = []
            used_width = 0
            added_width = width
        current.append(index)
        used_width += added_width
    if current:
        pages.append(tuple(current))
    return pages


GALLERY_CSS = """
.review-gallery { color: var(--st-text-color); font-family: var(--st-font); }
.review-info {
    min-height: 2.5em; margin-bottom: 8px; font-size: 0.9rem;
    overflow-wrap: anywhere; line-height: 1.45;
}
.review-scroll {
    max-height: 500px; overflow: auto; border: 1px solid var(--st-border-color);
    border-radius: 8px; padding: 6px; box-sizing: border-box;
}
.review-strip { display: flex; gap: 4px; align-items: flex-start; width: max-content; }
.review-frame {
    flex: 0 0 auto; display: block; padding: 0; margin: 0;
    border: 0; border-radius: 2px; background: transparent; cursor: pointer;
    color: inherit; line-height: 0;
}
.review-frame img { display: block; max-width: none; border-radius: 2px; }
.review-frame[aria-pressed="true"] { outline: 2px solid var(--st-primary-color); outline-offset: 1px; }
.review-frame:focus-visible { outline: 3px solid var(--st-primary-color); outline-offset: 1px; }
.review-frame:hover { filter: brightness(1.08); }
.review-missing {
    width: 88px; min-height: 120px; padding: 8px; box-sizing: border-box;
    background: var(--st-secondary-background-color); font-size: 0.8rem; line-height: 1.4;
}
"""

GALLERY_JS = """
export default function(component) {
    const { data, parentElement, setStateValue } = component;
    const root = parentElement.querySelector('.review-gallery');
    const info = root.querySelector('.review-info');
    const viewport = root.querySelector('.review-scroll');
    const strip = root.querySelector('.review-strip');
    const scrollLeft = viewport.scrollLeft;
    const scrollTop = viewport.scrollTop;
    strip.replaceChildren();
    const frames = data.frames || [];
    let selected = frames.find(frame => frame.id === data.selected);
    const buttons = new Map();
    function inspect(frame, announce = false) {
        selected = frame;
        info.textContent = frame ? frame.info : 'Click an image to show its name, time point, and channel.';
        for (const [id, button] of buttons) {
            button.setAttribute('aria-pressed', String(frame && id === frame.id));
        }
        if (announce && frame) setStateValue('selected', frame.id);
    }
    for (const frame of frames) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'review-frame';
        button.dataset.frameId = frame.id;
        button.setAttribute('aria-label', 'Inspect ' + frame.info);
        button.setAttribute('aria-pressed', 'false');
        if (frame.src) {
            const image = document.createElement('img');
            image.src = frame.src;
            image.alt = frame.alt;
            image.width = frame.width;
            image.height = frame.height;
            image.draggable = false;
            button.appendChild(image);
        } else {
            const missing = document.createElement('div');
            missing.className = 'review-missing';
            missing.textContent = 'Image unavailable';
            button.appendChild(missing);
        }
        button.onclick = () => inspect(frame, true);
        button.onkeydown = event => {
            const position = frames.indexOf(frame);
            const offset = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
            if (offset) {
                event.preventDefault();
                event.stopPropagation();
                const next = frames[Math.max(0, Math.min(frames.length - 1, position + offset))];
                buttons.get(next.id)?.focus();
                inspect(next, true);
            }
        };
        buttons.set(frame.id, button);
        strip.appendChild(button);
    }
    inspect(selected);
    viewport.scrollLeft = scrollLeft;
    viewport.scrollTop = scrollTop;

    function reportWidth() {
        const style = getComputedStyle(viewport);
        const width = Math.max(1, Math.floor(
            viewport.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
        ));
        if (data.reportedWidth !== width) setStateValue('viewport_width', width);
    }
    reportWidth();
    if (typeof ResizeObserver === 'undefined') return;
    let resizeTimer;
    const observer = new ResizeObserver(() => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(reportWidth, 120);
    });
    observer.observe(viewport);
    return () => {
        clearTimeout(resizeTimer);
        observer.disconnect();
    };
}
"""

# AppTest creates a fresh runtime/registry for every run. Cache one renderer per
# runtime registry so production registers once while tests remain isolated.
_RENDERERS = WeakKeyDictionary()


def _gallery_component():
    from streamlit.components.v2.get_bidi_component_manager import get_bidi_component_manager

    manager = get_bidi_component_manager()
    renderer = _RENDERERS.get(manager)
    if renderer is None:
        renderer = components.component(
            "moma_image_review_strip",
            html=(
                '<section class="review-gallery">'
                '<div class="review-info" role="status" aria-live="polite"></div>'
                '<div class="review-scroll"><div class="review-strip"></div></div>'
                '</section>'
            ),
            css=GALLERY_CSS,
            js=GALLERY_JS,
        )
        _RENDERERS[manager] = renderer
    return renderer


def image_data_url(rgb: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _remember_image_selection() -> None:
    value = st.session_state.get("image_review_gallery", {})
    selected = value.get("selected")
    snapshot = st.session_state.get("image_review_snapshot")
    if isinstance(selected, str) and snapshot is not None:
        if selected.isdigit() and 0 <= int(selected) < len(snapshot.frames):
            st.session_state.image_review_selected_frame = selected


def _remember_viewport_width() -> None:
    value = st.session_state.get("image_review_gallery", {})
    width = value.get("viewport_width")
    if isinstance(width, (int, float)) and np.isfinite(width) and width >= 1:
        st.session_state.image_review_viewport_width = int(width)


def render_image_gallery(frames: list[dict], reported_width: int | None = None) -> None:
    """Mount images and report the usable strip width for responsive pages."""

    _gallery_component()(
        data={
            "frames": frames,
            "selected": st.session_state.get("image_review_selected_frame"),
            "reportedWidth": reported_width,
        },
        key="image_review_gallery",
        default={"selected": None, "viewport_width": None},
        on_selected_change=_remember_image_selection,
        on_viewport_width_change=_remember_viewport_width,
        width="stretch",
        height="content",
    )
