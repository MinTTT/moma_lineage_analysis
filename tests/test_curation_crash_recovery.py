"""SQLite must recover the last committed curation after abrupt process death."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys

import pandas as pd

from moma_lineage_analysis.curation_store import (
    DB_FILENAME, create_dataset, fingerprint_sources, load_dataset,
)


def test_process_death_rolls_back_partial_label_batch(tmp_path: Path):
    source = tmp_path / 'fov_0_cells_df.csv'
    raw = pd.DataFrame({'fov_cell_index':[0,1], 'time_h':[0.0,1.0], 'spine_length':[2.0,3.0]})
    raw.to_csv(source, index=False)
    key = 'FOV_0_CH_1'
    entry = {'trajectory':raw,'div_indices':[],'original_div_indices':[],
             'excluded_indices':set(),'source_csv_path':str(source)}
    labels = raw.assign(trajectory_key=key, division_point=False, excluded_point=False,
                        effective_division_point=False, cell_cycle_index=0)
    db = tmp_path / DB_FILENAME
    create_dataset(db, trajectory_dict={key:entry}, labeled_frames={key:labels},
                   selected_keys={key}, sources=fingerprint_sources([source]),
                   settings={}, logic_version='2.3', current_key=key)
    program = '''
import os,sys
import moma_lineage_analysis.curation_store as store
path=sys.argv[1]
payload=store.load_dataset(path,1)
key='FOV_0_CH_1'
entry=payload['trajectory_dict'][key]
entry['excluded_indices']={1}
entry['_review_status']='in_progress'
labels=entry['trajectory'].assign(trajectory_key=key, division_point=False,
    excluded_point=[False,True],effective_division_point=False,cell_cycle_index=0)
# Kill the process after label UPDATEs but before transaction commit.
store._activate=lambda *args: os._exit(73)
store.save_changes(path,1,expected_revision=0,trajectory_dict={key:entry},
    labeled_frames={key:labels},selected_keys={key},current_key=key)
'''
    completed = subprocess.run([sys.executable,'-c',program,str(db)], timeout=30)
    assert completed.returncode == 73
    recovered = load_dataset(db,1)
    assert recovered['revision'] == 0
    assert recovered['trajectory_dict'][key]['excluded_indices'] == set()
    with closing(sqlite3.connect(db)) as connection:
        assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert connection.execute('SELECT sum(excluded_point) FROM selected_labeled').fetchone()[0] == 0
