"""Preview a training-ready version without mutating measurements."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
from . import store


def preview_finalization(db: str | Path, lineage_id: str) -> dict[str, Any]:
    snapshot = store.read_snapshot(db, lineage_id)
    frame = snapshot['frame']
    dropped = []
    for key, group in frame.groupby('trajectory_key', sort=False):
        flags = group.sort_values('time_h', kind='stable')['excluded_point'].tolist()
        run = longest = 0
        for flag in flags:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        if longest >= 5:
            dropped.append(str(key))
    retained = frame.loc[~frame.trajectory_key.isin(dropped)]
    return {'revision': snapshot['revision'], 'lineage_id': lineage_id,
            'trajectories_before': int(frame.trajectory_key.nunique()),
            'trajectories_retained': int(retained.trajectory_key.nunique()),
            'rows_before': len(frame), 'rows_retained': len(retained),
            'division_points': int(retained.division_point.sum()),
            'excluded_points': int(retained.excluded_point.sum()),
            'overlap_rows': int((retained.division_point & retained.excluded_point).sum()),
            'r004_dropped_keys': dropped}

