"""Shared visual style and navigation shortcuts for trajectory apps.

Constants and run splitting are shared by the lineage curation app and the
training-review app so both render the same trajectory style.
"""
from __future__ import annotations

from collections.abc import Mapping
import json

# High-contrast palette for cell-cycle segments (Tol "bright" extended with
# distinct hues). Neighboring cycles stay distinguishable for color-vision
# deficient users and colors never repeat within one trajectory.
TRAJECTORY_TRACE_COLORS = (
    '#0077BB', '#EE7733', '#009988', '#CC3311',
    '#33BBEE', '#EE3377', '#BBBB00', '#882255',
    '#44AA99', '#117733', '#AA4499', '#661100',
    '#6699CC', '#DDCC77', '#999933', '#332288',
)

# Semantic marker colors/sizes for review overlays. Exclusion markers stay
# small so they cannot cover neighboring observations of long trajectories.
APPLIED_EXCLUSION_MARKER = dict(symbol='x', size=7, color='#dc2626', line=dict(width=1.5))
AI_EXCLUSION_MARKER = dict(symbol='x-open', size=6, color='#f97316', line=dict(width=1))
AI_DIVISION_MARKER = dict(symbol='diamond-open', size=9, color='#7c3aed', line=dict(width=1.5))


def trajectory_plot_runs(trajectory, excluded_indices, boundary_indices=(), *, presorted=False):
    """Return (group, frame) runs split at exclusions and boundaries."""
    ordered = trajectory if presorted else trajectory.sort_values('time_h', kind='stable')
    excluded = set(excluded_indices)
    boundaries = set(boundary_indices)
    runs = []
    run_start = None
    boundary_group = 0
    for position, cell_id in enumerate(ordered['fov_cell_index'].tolist()):
        if cell_id in excluded:
            if run_start is not None:
                runs.append((boundary_group, ordered.iloc[run_start:position].copy()))
                run_start = None
            continue
        if run_start is None:
            run_start = position
        if cell_id in boundaries:
            runs.append((boundary_group, ordered.iloc[run_start:position + 1].copy()))
            run_start = None
            boundary_group += 1
    if run_start is not None:
        runs.append((boundary_group, ordered.iloc[run_start:].copy()))
    return runs


def shortcut_script(bindings: Mapping[str, str]) -> str:
    """Build the shared keydown handler used by every trajectory app.

    bindings maps a key ('ArrowLeft', 'd', ...) to button text that must be
    present. Only visible, enabled actions respond. Text fields, selects,
    sliders, editors and dialogs keep their own keys; modifiers are ignored.
    """
    if not bindings:
        raise ValueError('At least one keyboard binding is required')
    normalized = {}
    for key, label in bindings.items():
        if not isinstance(key, str) or not key or not isinstance(label, str) or not label:
            raise ValueError('Keyboard bindings require non-empty string keys and labels')
        normalized[key.lower() if len(key) == 1 else key] = label
    encoded = json.dumps(normalized, ensure_ascii=True).replace('</', '<\\/')
    return f"""<script>
    (() => {{
      const host = window.parent;
      const doc = window.parent.document;
      const handlerName = '__momaTrajectoryKeyHandler';
      const bindings = {encoded};
      if (host[handlerName]) {{ doc.removeEventListener('keydown', host[handlerName]); }}
      function visible(button) {{
        if (!button.getClientRects().length) return false;
        for (let node = button; node; node = node.parentElement) {{
          if (node.hidden || node.hasAttribute('inert') || node.getAttribute('aria-hidden') === 'true') return false;
          const style = host.getComputedStyle(node);
          if (style.display === 'none' || style.visibility === 'hidden' || style.visibility === 'collapse') return false;
          if (node.tagName === 'DETAILS' && !node.open) return false;
        }}
        return true;
      }}
      host[handlerName] = function(e) {{
        if (e.defaultPrevented || e.isComposing || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
        if (Array.from(doc.querySelectorAll('[role="dialog"]')).some(visible)) return;
        const target = e.target;
        if (!target || target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)
            || target.closest('[contenteditable="true"], [contenteditable=""], [role="slider"]')
            || target.closest('[role="grid"], [role="combobox"], [role="listbox"], '
                              + '[data-testid="stDataFrame"], [data-testid="stDataEditor"]')) return;
        const label = bindings[e.key] ?? bindings[e.key.toLowerCase()];
        if (!label) return;
        const button = Array.from(doc.querySelectorAll('button')).find(button =>
          button.innerText.includes(label) && !button.disabled && visible(button));
        if (button) {{
          e.preventDefault();
          button.click();
        }}
      }};
      doc.addEventListener('keydown', host[handlerName]);
    }})();
    </script>"""
