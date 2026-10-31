"""Reviewed datasets remain verifiable and historical imports cannot undo deletion."""
import importlib.util
import sqlite3
from pathlib import Path
import pytest
from moma_lineage_analysis.trajectory_cleaning import dataset_store as ds
from moma_lineage_analysis.trainset_review import store


def fixture(tmp_path):
    spec = importlib.util.spec_from_file_location('review_fixture', Path(__file__).with_name('test_store.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module._build_store(tmp_path)


def test_working_hash_corruption_is_detected(tmp_path):
    f = fixture(tmp_path); db = f['db']
    store.delete_trajectory(db, 't1', expected_revision=0, confirmation_key='t1')
    with ds.connect_readonly(db) as conn:
        assert ds.verify_store(conn, repo_root=f['root'])['status'] == 'PASS'
    with sqlite3.connect(db) as conn:
        conn.execute('UPDATE observation SET spine_length=spine_length+0.25 WHERE trajectory_id=(SELECT trajectory_id FROM trajectory WHERE trajectory_key=?)', ('t2',))
    with ds.connect_readonly(db) as conn:
        result = ds.verify_store(conn, repo_root=f['root'])
    assert result['status'] == 'FAIL'
    assert any('hash mismatch' in message or 'working version' in message for message in result['failures'])


def test_finalized_native_pin_requires_matching_audit_proof(tmp_path):
    f = fixture(tmp_path); db = f['db']
    result = store.finalize(db, f['lineage_a'], 'review_approved', expected_revision=0)
    with ds.connect_readonly(db) as conn:
        assert ds.verify_store(conn, repo_root=f['root'], export_dir=tmp_path/'export')['status'] == 'PASS'
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM review_event WHERE event_type='finalize'")
    with ds.connect_readonly(db) as conn:
        assert ds.verify_store(conn, repo_root=f['root'])['status'] == 'FAIL'


def test_build_import_cannot_resurrect_deleted_trajectories(tmp_path):
    f = fixture(tmp_path); db = f['db']
    store.delete_trajectory(db, 't1', expected_revision=0, confirmation_key='t1')
    with ds.connect(db) as conn:
        with pytest.raises(ValueError, match='prevent restoring permanently deleted'):
            ds.assert_import_allowed(conn)
        with pytest.raises(ValueError, match='Historical CSV'):
            ds.import_corpus(conn, repo_root=f['root'], csv_path=f['root']/'corpora/corpus_a.csv', dataset_id='reimport', kind='batch', label='test', round_name=None, verification_basis='test', human_verified=True)
    with pytest.raises(ValueError):
        store.load_trajectory(db, 't1')
