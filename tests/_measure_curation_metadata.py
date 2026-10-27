"""Measure metadata normalization with identical real measurements in temp DBs."""
from contextlib import closing
import json
from pathlib import Path
import tempfile
import sqlite3
import numpy as np
import pandas as pd
from moma_lineage_analysis import curation_store as store
from moma_lineage_analysis.moma_logic import process_channel_worker
from moma_lineage_analysis.trajectory_cleaning.streamlit_api import resolve_division_exclusion_conflicts

root = Path('tests/data/exported_example')
entries, labels = {}, {}
for source in sorted(root.glob('*_cells_df.csv')):
    frame = pd.read_csv(source)
    frame['fov_cell_index'] = np.arange(len(frame))
    for channel, rows in frame.groupby('channels', sort=False):
        result = process_channel_worker(source.stem.split('_')[1], channel, rows.copy(),
                                        1.2, 's', 'longest_lineage', 0.0, None, None,
                                        min_spine_length=1.0)
        if not result:
            continue
        key, entry = result
        entry['source_csv_path'] = str(source.resolve())
        entry['excluded_indices'] = set()
        resolution = resolve_division_exclusion_conflicts(entry['trajectory'], entry['div_indices'], set())
        labeled = entry['trajectory'].sort_values('time_h', kind='stable').copy()
        labeled['trajectory_key'] = key
        labeled['division_point'] = labeled.fov_cell_index.isin(entry['div_indices'])
        labeled['excluded_point'] = False
        labeled['effective_division_point'] = labeled.fov_cell_index.isin(resolution.effective_division_indices)
        labeled['cell_cycle_index'] = labeled.effective_division_point.shift(1, fill_value=False).cumsum().astype(int)
        entries[key], labels[key] = entry, labeled

with tempfile.TemporaryDirectory(prefix='moma-metadata-benchmark-') as directory:
    new = Path(directory) / 'v3.sqlite3'
    old = Path(directory) / 'v2.sqlite3'
    store.create_dataset(new, trajectory_dict=entries, labeled_frames=labels, selected_keys=set(entries),
                         sources=store.fingerprint_sources(sorted(root.glob('*_cells_df.csv'))), settings={},
                         logic_version='2.3', current_key=next(iter(entries)))
    with closing(sqlite3.connect(new)) as connection, closing(sqlite3.connect(old)) as legacy:
        connection.backup(legacy)
    with closing(sqlite3.connect(old)) as connection:
        views = connection.execute("SELECT name,sql FROM sqlite_master WHERE type='view'").fetchall()
        for name, _ in views:
            connection.execute(f'DROP VIEW {name}')
        connection.execute("""CREATE TABLE trajectories_v2 (dataset_id INTEGER NOT NULL REFERENCES datasets(dataset_id), trajectory_key TEXT NOT NULL, trajectory_order INTEGER NOT NULL, source_filename TEXT NOT NULL, source_path TEXT NOT NULL, selected INTEGER NOT NULL CHECK(selected IN (0,1)), review_status TEXT NOT NULL CHECK(review_status IN ('reviewed','unreviewed','in_progress')), reviewed_at TEXT, original_div_indices_json TEXT NOT NULL, dtype_json TEXT NOT NULL, PRIMARY KEY(dataset_id,trajectory_key))""")
        for row in connection.execute('SELECT t.*,s.filename,s.path,m.dtype_json FROM trajectories t JOIN source_files s ON s.dataset_id=t.dataset_id AND s.source_id=t.source_id JOIN measurement_schemas m ON m.schema_id=t.schema_id').fetchall():
            dataset, key, order, source_id, schema_id, selected, status, reviewed, filename, path, dtype = row
            connection.execute('INSERT INTO trajectories_v2 VALUES (?,?,?,?,?,?,?,?,?,?)',
                               (dataset,key,order,filename,path,selected,status,reviewed,
                                store._json(entries[key]['original_div_indices']),dtype))
        connection.execute('DROP TABLE trajectories')
        connection.execute('ALTER TABLE trajectories_v2 RENAME TO trajectories')
        connection.execute('CREATE TABLE source_files_v2 (dataset_id INTEGER NOT NULL REFERENCES datasets(dataset_id), filename TEXT NOT NULL, sha256 TEXT NOT NULL, size INTEGER NOT NULL, path TEXT, PRIMARY KEY(dataset_id,filename))')
        connection.execute('INSERT INTO source_files_v2 SELECT dataset_id,filename,sha256,size,path FROM source_files')
        connection.execute('DROP TABLE source_files')
        connection.execute('ALTER TABLE source_files_v2 RENAME TO source_files')
        connection.execute('DROP TABLE measurement_schemas')
        obs_schema = next(statement for statement in store._SCHEMA if statement.startswith('CREATE TABLE observations'))
        obs_schema = obs_schema.replace('CREATE TABLE observations', 'CREATE TABLE observations_v2').replace('_moma_division_state INTEGER NOT NULL DEFAULT 0 CHECK(_moma_division_state >= 0)', '_moma_division_point INTEGER NOT NULL DEFAULT 0 CHECK(_moma_division_point IN (0,1))')
        connection.execute(obs_schema)
        mapping = store._column_map(connection)
        for physical in mapping.values():
            connection.execute(f'ALTER TABLE observations_v2 ADD COLUMN {store._quote(physical)}')
        columns = [row[1] for row in connection.execute('PRAGMA table_info(observations_v2)')]
        projection = ['(_moma_division_state & 1)' if name == '_moma_division_point' else store._quote(name) for name in columns]
        connection.execute('INSERT INTO observations_v2 SELECT ' + ','.join(projection) + ' FROM observations')
        connection.execute('DROP TABLE observations')
        connection.execute('ALTER TABLE observations_v2 RENAME TO observations')
        for _, sql in views:
            connection.execute(sql.replace('(o._moma_division_state & 1)', 'o._moma_division_point'))
        connection.execute('PRAGMA user_version=2')
        connection.commit()
        connection.execute('VACUUM')
        assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert not connection.execute('PRAGMA foreign_key_check').fetchall()
        old_metadata = connection.execute('SELECT sum(length(source_filename)+length(source_path)+length(original_div_indices_json)+length(dtype_json)) FROM trajectories').fetchone()[0]
    with closing(sqlite3.connect(new)) as connection:
        connection.execute('VACUUM')
        connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        schema_count = connection.execute('SELECT count(*) FROM measurement_schemas').fetchone()[0]
        schema_bytes = connection.execute('SELECT sum(length(dtype_json)) FROM measurement_schemas').fetchone()[0]
    for name, path in [('before',old),('after',new)]:
        with closing(sqlite3.connect(path)) as connection:
            result = pd.read_sql_query('SELECT * FROM selected_labeled', connection)
        if name == 'before':
            expected = result
        else:
            pd.testing.assert_frame_equal(result,expected)
    print(json.dumps({'trajectories':len(entries),'observations':sum(len(e['trajectory']) for e in entries.values()),
                      'before_v2_bytes':old.stat().st_size,'after_v3_bytes':new.stat().st_size,
                      'saved_bytes':old.stat().st_size-new.stat().st_size,
                      'saved_percent':round((1-new.stat().st_size/old.stat().st_size)*100,2),
                      'old_repeated_metadata_text_bytes':old_metadata,'shared_dtype_schemas':schema_count,
                      'shared_dtype_text_bytes':schema_bytes,'sql_view_exact_parity':'pass'}))
