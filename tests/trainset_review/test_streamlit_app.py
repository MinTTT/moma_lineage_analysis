"""Focused integration checks for the separate training-dataset review app."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from moma_lineage_analysis.paths import PACKAGE_ROOT
from moma_lineage_analysis.trainset_review import store
from moma_lineage_analysis.trajectory_cleaning import dataset_store as ds

APP = str(PACKAGE_ROOT / 'trainset_review/streamlit_app.py')


def fixture_store(tmp_path):
    spec = importlib.util.spec_from_file_location('fixture_store', Path(__file__).with_name('test_store.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._build_store(tmp_path)


def element(elements, label):
    # AppTest can retain the preceding rerun's nodes after a success message
    # shifts the layout. Interact with the most recently rendered widget.
    return next(e for e in reversed(list(elements)) if e.label == label)


def select_ids(app, ids):
    app.session_state['review_selected_ids'] = [str(int(value)) for value in ids]
    app.run()


def app_for(monkeypatch, db):
    monkeypatch.setenv('MOMA_TRAINSET_STORE', str(db))
    return AppTest.from_file(APP, default_timeout=45).run()


def test_review_and_delete_are_persistent_and_affected_versions_invalidated(tmp_path, monkeypatch):
    fixture = fixture_store(tmp_path)
    db = fixture['db']
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    app = app_for(monkeypatch, db)
    assert not app.exception
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    element(app.selectbox, 'Dataset version').set_value('corpus_a@v1').run()
    element(app.selectbox, 'Trajectory').set_value('t1').run()
    select_ids(app, ['1'])
    element(app.button, 'Add division [D]').click().run()
    assert not app.exception
    assert bool(store.load_trajectory(db, 't1').set_index('fov_cell_index').loc[1, 'division_point'])
    element(app.button, 'Undo last label edit [U]').click().run()
    assert not app.exception
    assert not bool(store.load_trajectory(db, 't1').set_index('fov_cell_index').loc[1, 'division_point'])
    element(app.button, 'Mark reviewed [K]').click().run()
    assert not app.exception
    assert store.review_status(db, 't1')['reviewed']
    assert not any(i.label == 'Type the full trajectory key to delete' for i in app.text_input)
    element(app.button, 'Remove trajectory from database [Delete]').click().run()
    assert not app.exception and not app.error
    assert element(app.selectbox, 'Trajectory').value == 't2'
    assert any('Trajectory t1 removed' in item.value for item in app.toast)
    with pytest.raises(ValueError):
        store.load_trajectory(db, 't1')
    assert all(not v['frozen'] for v in store.list_versions(db) if v['lineage_id'] in ('corpus_a@v1', 'corpus_b@v1'))
    history = store.read_history(db, 't1')
    snapshots = [event for event in history if event.get('snapshot_path')]
    assert snapshots and all(Path(event['snapshot_path']).is_file() for event in snapshots)


def test_stale_page_is_blocked_until_explicit_reload(tmp_path, monkeypatch):
    fixture = fixture_store(tmp_path)
    db = fixture['db']
    app = app_for(monkeypatch, db)
    assert not app.exception
    element(app.selectbox, 'Dataset version').set_value('corpus_a@v1').run()
    element(app.selectbox, 'Trajectory').set_value('t1').run()
    store.mark_reviewed(db, 't2', True, expected_revision=store.revision(db))
    app.run()
    assert not app.exception
    assert any('changed in another session' in warning.value for warning in app.warning)
    assert not any(b.label == 'Add division [D]' for b in app.button)
    element(app.button, 'Reload changed dataset').click().run()
    assert not app.exception
    assert any(b.label == 'Add division [D]' for b in app.button)


def test_finalize_and_native_store_verification(tmp_path, monkeypatch):
    fixture = fixture_store(tmp_path)
    db = fixture['db']
    app = app_for(monkeypatch, db)
    assert not app.exception
    element(app.selectbox, 'Dataset version').set_value('corpus_a@v1').run()
    before_versions = {v['lineage_id'] for v in store.list_versions(db)}
    element(app.checkbox, 'These labels are final for training').check().run()
    element(app.button, 'Make training-ready [T]').click().run()
    assert not app.exception
    created = [v['lineage_id'] for v in store.list_versions(db) if v['lineage_id'] not in before_versions]
    assert len(created) == 1
    new_lineage = created[0]
    assert new_lineage.startswith('reviewed_')
    with ds.connect_readonly(db) as conn:
        frame, info = ds.load_training_frame(conn, [new_lineage], require_human_verified=True)
        assert frame.trajectory_key.nunique() == 2  # t_long omitted only from new membership.
        result = ds.verify_store(conn, repo_root=fixture['root'], export_dir=tmp_path/'exports')
        assert result['status'] == 'PASS', result['failures']
    assert len(store.load_trajectory(db, 't_long')) == 6


def test_r015_preview_and_evaluate_filters(tmp_path, monkeypatch):
    fixture = fixture_store(tmp_path)
    db = fixture['db']
    app = app_for(monkeypatch, db)
    from moma_lineage_analysis.model_registry import R015_CNN_TAG
    element(app.selectbox, 'Model checkpoint').set_value(R015_CNN_TAG).run()
    element(app.selectbox, 'Dataset version').set_value('corpus_a@v1').run()
    element(app.selectbox, 'Trajectory').set_value('t1').run()
    element(app.button, 'Run model preview [P]').click().run()
    assert not app.exception
    assert 'review_predictions' in app.session_state
    element(app.button, 'Evaluate filtered trajectories [V]').click().run()
    assert not app.exception
    assert any(m.label == 'Exclusion F1' for m in app.metric)
    element(app.selectbox, 'Queue order').set_value('Model disagreement').run()
    assert not app.exception


@pytest.mark.parametrize('removed_key, next_key', [('t2', 't_long'), ('t_long', 't2')])
def test_one_click_remove_advances_in_queue_and_survives_new_session(tmp_path, monkeypatch,
                                                                   removed_key, next_key):
    fixture = fixture_store(tmp_path)
    db = fixture['db']
    before = {key: store.load_trajectory(db, key) for key in ('t1', 't2', 't_long', 't3')}
    app = app_for(monkeypatch, db)
    element(app.selectbox, 'Dataset version').set_value('corpus_a@v1').run()
    element(app.selectbox, 'Trajectory').set_value(removed_key).run()
    element(app.button, 'Remove trajectory from database [Delete]').click().run()
    assert not app.exception and not app.error
    assert element(app.selectbox, 'Trajectory').value == next_key
    assert store.revision(db) == 1  # One click performs exactly one mutation.
    with pytest.raises(ValueError):
        store.load_trajectory(db, removed_key)
    import pandas as pd
    for key, frame in before.items():
        if key != removed_key:
            pd.testing.assert_frame_equal(frame, store.load_trajectory(db, key))
    reopened = app_for(monkeypatch, db)
    element(reopened.selectbox, 'Dataset version').set_value('corpus_a@v1').run()
    assert not reopened.exception
    assert removed_key not in element(reopened.selectbox, 'Trajectory').options


def test_one_click_remove_last_filtered_trajectory_shows_empty_queue(tmp_path, monkeypatch):
    fixture = fixture_store(tmp_path)
    db = fixture['db']
    app = app_for(monkeypatch, db)
    element(app.selectbox, 'Dataset version').set_value('corpus_a@v1').run()
    element(app.text_input, 'Find trajectory').set_value('t1').run()
    element(app.button, 'Remove trajectory from database [Delete]').click().run()
    assert not app.exception and not app.error
    assert any('No trajectories match' in message.value for message in app.info)
    assert not any(b.label == 'Remove trajectory from database [Delete]' for b in app.button)
    assert store.revision(db) == 1
    assert 't1' not in store.load_frame(db, 'corpus_a@v1').trajectory_key.tolist()
    assert 't1' not in store.load_frame(db, 'corpus_b@v1').trajectory_key.tolist()
    element(app.text_input, 'Find trajectory').set_value('').run()
    assert element(app.selectbox, 'Trajectory').value == 't2'


def test_stale_remove_click_does_not_delete_current_trajectory(tmp_path, monkeypatch):
    fixture = fixture_store(tmp_path)
    db = fixture['db']
    app = app_for(monkeypatch, db)
    element(app.selectbox, 'Dataset version').set_value('corpus_a@v1').run()
    element(app.selectbox, 'Trajectory').set_value('t1').run()
    store.mark_reviewed(db, 't2', True, expected_revision=0)
    element(app.button, 'Remove trajectory from database [Delete]').click().run()
    assert not app.exception
    assert any('changed in another session' in message.value for message in app.warning)
    assert store.revision(db) == 1
    assert len(store.load_trajectory(db, 't1')) == 4
    assert not any(e['event_type'] == 'delete_trajectory' for e in store.read_history(db))


def test_failed_remove_keeps_current_trajectory_and_queue(tmp_path, monkeypatch):
    fixture = fixture_store(tmp_path)
    db = fixture['db']
    app = app_for(monkeypatch, db)
    element(app.selectbox, 'Dataset version').set_value('corpus_a@v1').run()
    element(app.selectbox, 'Trajectory').set_value('t1').run()
    def failed_backup(*args, **kwargs):
        raise OSError('Snapshot failed')
    monkeypatch.setattr(store, 'delete_trajectory', failed_backup)
    element(app.button, 'Remove trajectory from database [Delete]').click().run()
    assert not app.exception
    assert any('Snapshot failed' in message.value for message in app.error)
    assert element(app.selectbox, 'Trajectory').value == 't1'
    assert store.revision(db) == 0
    assert len(store.load_trajectory(db, 't1')) == 4
