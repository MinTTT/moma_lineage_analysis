"""Autosave/recovery integration through the real curation app."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from contextlib import closing
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from moma_lineage_analysis import moma_logic
from moma_lineage_analysis.curation_store import DB_FILENAME, fingerprint_sources, load_dataset
from moma_lineage_analysis.app.curation_session import initialize_curation
from .test_streamlit_division_resolution_panel import (
    APP_FILE, _app_with_entries, _element, _select_plot_points, _trajectory_entry,
)


def _fixture(directory: Path):
    source = directory / "fov_0_cells_df.csv"
    first = _trajectory_entry()
    second = _trajectory_entry()
    first['trajectory']['channels'] = 1
    second['trajectory']['channels'] = 2
    for entry in (first, second):
        entry['trajectory']['time_s'] = entry['trajectory']['time_h'] * 3600
        entry['trajectory']['time_point'] = [f't{index}.tiff' for index in range(8)]
        entry['trajectory']['cell_index'] = 1
        entry['source_csv_path'] = str(source)
        entry['_review_status'] = 'unreviewed'
    pd.concat([first['trajectory'], second['trajectory']]).to_csv(source, index=False)
    return {'FOV_0_CH_1': first, 'FOV_0_CH_2': second}, source


def _labels(entry, key):
    # Initial fixture has clean baseline divisions; the live app later provides
    # its authoritative conflict-resolved CSV builder for every edit.
    frame = entry['trajectory'].sort_values('time_h', kind='stable').copy()
    frame['trajectory_key'] = key
    frame['division_point'] = frame['fov_cell_index'].isin(entry['div_indices'])
    frame['excluded_point'] = frame['fov_cell_index'].isin(entry['excluded_indices'])
    frame['effective_division_point'] = frame['division_point']
    frame['cell_cycle_index'] = frame['effective_division_point'].shift(1, fill_value=False).cumsum().astype(int)
    return frame


def _saved_app(directory):
    entries, source = _fixture(directory)
    state = {'trajectory_dict': entries, 'selected_keys': set(entries), 'current_idx': 0}
    initialize_curation(
        state, directory / DB_FILENAME, sources=fingerprint_sources([source]),
        settings={'time_unit':'s', 'div_thresh':1.2, 'mode':'longest_lineage',
                  'min_duration':0.0, 'exclude_mode':True, 'min_spine':1.0,
                  'enable_start':False, 'start_time':None, 'enable_stop':False, 'stop_time':None},
        logic_version=str(moma_logic.LOGIC_VERSION), builder=_labels,
    )
    app = _app_with_entries(state['trajectory_dict'])
    for key, value in state.items():
        app.session_state[key] = value
    app.run()
    assert not app.exception, app.exception
    return app


class CurationAutosaveTests(unittest.TestCase):
    def test_manual_edits_review_confirmation_drop_and_fresh_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            app = _saved_app(directory)
            _select_plot_points(app, [{'customdata':[2]}])
            _element(app.button, '🚫 Exclude Selected').click().run()
            self.assertFalse(app.exception)
            saved = load_dataset(directory / DB_FILENAME, 1)
            self.assertEqual(saved['trajectory_dict']['FOV_0_CH_1']['excluded_indices'], {2})
            self.assertEqual(saved['trajectory_dict']['FOV_0_CH_1']['_review_status'], 'in_progress')
            _element(app.button, 'Next ➡️').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['current_idx'], 1)
            saved = load_dataset(directory / DB_FILENAME, 1)
            self.assertEqual(saved['trajectory_dict']['FOV_0_CH_1']['_review_status'], 'reviewed')
            self.assertEqual(saved['current_key'], 'FOV_0_CH_2')
            app.button(key='toggle_trajectory_selection_btn').click().run()
            saved = load_dataset(directory / DB_FILENAME, 1)
            self.assertEqual(saved['selected_keys'], {'FOV_0_CH_1'})
            with closing(sqlite3.connect(directory / DB_FILENAME)) as connection:
                count = connection.execute('SELECT count(*) FROM reviewed_selected_labeled').fetchone()[0]
            self.assertEqual(count, 8)

            fresh = AppTest.from_file(APP_FILE, default_timeout=30).run()
            _element(fresh.sidebar.text_input, 'Data Directory Path').set_value(str(directory)).run()
            fresh.button(key='resume_curation').click().run()
            self.assertFalse(fresh.exception)
            self.assertEqual(fresh.session_state['current_idx'], 1)
            self.assertEqual(fresh.session_state['selected_keys'], {'FOV_0_CH_1'})
            self.assertEqual(fresh.session_state['trajectory_dict']['FOV_0_CH_1']['excluded_indices'], {2})
            self.assertEqual(_element(fresh.sidebar.selectbox, 'Time Unit').value, 's')

    def test_failed_commit_retains_edits_and_blocks_navigation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            app = _saved_app(directory)
            with patch('moma_lineage_analysis.app.curation_session.save_changes', side_effect=OSError('disk full')):
                _select_plot_points(app, [{'customdata':[2]}])
                _element(app.button, '🚫 Exclude Selected').click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state['trajectory_dict']['FOV_0_CH_1']['excluded_indices'], {2})
                self.assertIn('disk full', app.session_state['curation_error'])
                _element(app.button, 'Next ➡️').click().run()
                self.assertEqual(app.session_state['current_idx'], 0)
            app.button(key='retry_curation_save').click().run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state['curation_error'])
            saved = load_dataset(directory / DB_FILENAME, 1)
            self.assertEqual(saved['trajectory_dict']['FOV_0_CH_1']['excluded_indices'], {2})

    def test_reviewed_edit_revokes_confirmation_and_view_matches_csv_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            app = _saved_app(directory)
            _element(app.button, 'Next ➡️').click().run()
            _element(app.button, '⬅️ Prev').click().run()
            _select_plot_points(app, [{'customdata':[2]}])
            _element(app.button, '🚫 Exclude Selected').click().run()
            self.assertFalse(app.exception)
            saved = load_dataset(directory / DB_FILENAME, 1)
            self.assertEqual(saved['trajectory_dict']['FOV_0_CH_1']['_review_status'], 'in_progress')
            with closing(sqlite3.connect(directory / DB_FILENAME)) as connection:
                self.assertEqual(connection.execute('SELECT count(*) FROM reviewed_selected_labeled').fetchone()[0], 0)
                actual = pd.read_sql_query('SELECT * FROM selected_labeled ORDER BY trajectory_key,time_h', connection)
            _element(app.sidebar.text_input, 'Data Directory Path').set_value(temporary).run()
            _element(app.sidebar.button, '📊 Export with Labels').click().run()
            self.assertFalse(app.exception)
            expected = pd.read_csv(directory / 'selected_labeled.csv').sort_values(['trajectory_key','time_h']).reset_index(drop=True)
            for column in ('division_point','excluded_point','effective_division_point'):
                expected[column] = expected[column].astype(int)
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    def test_settings_change_creates_new_version_and_source_change_blocks_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            app = _saved_app(directory)
            _element(app.sidebar.text_input, 'Data Directory Path').set_value(temporary).run()
            _element(app.sidebar.number_input, 'Min Duration (h)').set_value(0.0).run()
            _element(app.sidebar.number_input, 'Division Threshold').set_value(1.3).run()
            _element(app.sidebar.slider, 'Number of Cores').set_value(1).run()
            _element(app.sidebar.button, 'Process Data').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['curation_dataset_id'], 2)
            source = directory / 'fov_0_cells_df.csv'
            source.write_text(source.read_text() + '\n', encoding='utf-8')
            fresh = AppTest.from_file(APP_FILE, default_timeout=30).run()
            _element(fresh.sidebar.text_input, 'Data Directory Path').set_value(temporary).run()
            fresh.button(key='resume_curation').click().run()
            self.assertFalse(fresh.exception)
            self.assertFalse(fresh.session_state['trajectory_dict'])
            self.assertTrue(any('changed' in error.value for error in fresh.sidebar.error))

    def test_ai_apply_undo_and_fov_drop_are_persisted(self):
        from .test_streamlit_division_resolution_panel import _save_tiny_checkpoint, _load_tiny_ai_model

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            app = _saved_app(directory)
            checkpoint = directory / 'model.pt'
            _save_tiny_checkpoint(str(checkpoint))
            _load_tiny_ai_model(app, str(checkpoint))
            _element(app.button, '✨ Run AI Preview').click().run()
            _element(app.button, '✅ Apply AI Suggestions').click().run()
            self.assertFalse(app.exception)
            saved = load_dataset(directory / DB_FILENAME, 1)
            current = app.session_state['trajectory_dict']['FOV_0_CH_1']
            self.assertEqual(saved['trajectory_dict']['FOV_0_CH_1']['excluded_indices'], current['excluded_indices'])
            self.assertEqual(set(saved['trajectory_dict']['FOV_0_CH_1']['div_indices']), set(current['div_indices']))
            _element(app.button, '↩️ Undo AI Changes').click().run()
            self.assertFalse(app.exception)
            saved = load_dataset(directory / DB_FILENAME, 1)
            self.assertEqual(saved['trajectory_dict']['FOV_0_CH_1']['excluded_indices'], set())
            app.button(key='drop_fov_trajectories').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(load_dataset(directory / DB_FILENAME, 1)['selected_keys'], set())

    def test_revision_conflict_retains_unsaved_labels_and_blocks_mark_next(self):
        from moma_lineage_analysis.curation_store import save_changes

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            app = _saved_app(directory)
            revision = app.session_state['curation_revision']
            save_changes(directory / DB_FILENAME, 1, expected_revision=revision,
                         trajectory_dict={}, labeled_frames={},
                         selected_keys={'FOV_0_CH_1','FOV_0_CH_2'}, current_key='FOV_0_CH_1')
            _select_plot_points(app, [{'customdata':[2]}])
            _element(app.button, '🚫 Exclude Selected').click().run()
            self.assertFalse(app.exception)
            self.assertIn('revision', app.session_state['curation_error'])
            _element(app.button, 'Next ➡️').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['current_idx'], 0)
            self.assertEqual(app.session_state['trajectory_dict']['FOV_0_CH_1']['excluded_indices'], {2})
            saved = load_dataset(directory / DB_FILENAME, 1)
            self.assertEqual(saved['trajectory_dict']['FOV_0_CH_1']['excluded_indices'], set())

    def test_export_includes_dropped_trajectory_fluorescence_columns_in_original_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            entries, source = _fixture(directory)
            # The dropped second trajectory owns a fluorescence channel absent
            # from the selected first trajectory; export schema still retains it.
            entries['FOV_0_CH_1']['trajectory']['green'] = np.arange(8.0)
            entries['FOV_0_CH_2']['trajectory']['red'] = np.arange(8.0) * 10
            pd.concat([entries['FOV_0_CH_1']['trajectory'], entries['FOV_0_CH_2']['trajectory']]).to_csv(source, index=False)
            state = {'trajectory_dict': entries, 'selected_keys': {'FOV_0_CH_1'}, 'current_idx': 0}
            initialize_curation(
                state, directory / DB_FILENAME, sources=fingerprint_sources([source]),
                settings={'time_unit':'s'}, logic_version=str(moma_logic.LOGIC_VERSION), builder=_labels,
            )
            app = _app_with_entries(entries)
            for key, value in state.items(): app.session_state[key] = value
            app.session_state['selected_keys'] = {'FOV_0_CH_2', 'FOV_0_CH_1'}
            _element(app.sidebar.text_input, 'Data Directory Path').set_value(temporary).run()
            _element(app.sidebar.button, '📊 Export with Labels').click().run()
            self.assertFalse(app.exception)
            exported = pd.read_csv(directory / 'selected_labeled.csv')
            self.assertEqual(list(exported['trajectory_key']), ['FOV_0_CH_1'] * 8 + ['FOV_0_CH_2'] * 8)
            with closing(sqlite3.connect(directory / DB_FILENAME)) as connection:
                actual = pd.read_sql_query('SELECT * FROM selected_labeled', connection)
            expected = exported.copy()
            for column in ('division_point','excluded_point','effective_division_point'):
                expected[column] = expected[column].astype(int)
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    def test_conflict_recovery_saves_local_copy_then_loads_latest_revision(self):
        from moma_lineage_analysis.curation_store import save_changes

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            app = _saved_app(directory)
            save_changes(directory / DB_FILENAME, 1, expected_revision=app.session_state['curation_revision'],
                         trajectory_dict={}, labeled_frames={},
                         selected_keys={'FOV_0_CH_1','FOV_0_CH_2'}, current_key='FOV_0_CH_1')
            _select_plot_points(app, [{'customdata':[2]}])
            _element(app.button, '🚫 Exclude Selected').click().run()
            self.assertIn('revision', app.session_state['curation_error'])
            app.button(key='curation_conflict_recover').click().run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state['curation_error'])
            copies = list(directory.glob('moma_curation_recovery_*.sqlite3'))
            self.assertEqual(len(copies), 1)
            recovered = load_dataset(copies[0], 1)
            self.assertEqual(recovered['trajectory_dict']['FOV_0_CH_1']['excluded_indices'], {2})
            saved = load_dataset(directory / DB_FILENAME, 1)
            self.assertEqual(saved['trajectory_dict']['FOV_0_CH_1']['excluded_indices'], set())

    def test_failed_mark_review_does_not_show_confirmed_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            app = _saved_app(directory)
            with patch('moma_lineage_analysis.app.curation_session.save_changes', side_effect=OSError('read-only')):
                _element(app.button, 'Next ➡️').click().run()
            self.assertFalse(app.exception)
            entry = app.session_state['trajectory_dict']['FOV_0_CH_1']
            self.assertEqual(entry['_review_status'], 'unreviewed')
            self.assertIsNone(entry['_reviewed_at'])
            self.assertEqual(app.session_state['current_idx'], 0)

    def test_save_current_session_preserves_old_session_as_new_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _saved_app(directory)
            entries, source = _fixture(directory)
            entries['FOV_0_CH_1']['excluded_indices'] = {2}
            for entry in entries.values():
                entry['original_div_indices'] = list(entry['div_indices'])
                entry['_review_status'] = 'in_progress'
            app = _app_with_entries(entries)
            _element(app.sidebar.text_input, 'Data Directory Path').set_value(temporary).run()
            app.button(key='migrate_curation_session').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['curation_dataset_id'], 2)
            saved = load_dataset(directory / DB_FILENAME, 2)
            self.assertEqual(saved['trajectory_dict']['FOV_0_CH_1']['excluded_indices'], {2})
            first_saved = load_dataset(directory / DB_FILENAME, 1)
            self.assertEqual(first_saved['trajectory_dict']['FOV_0_CH_1']['excluded_indices'], set())
            fresh = AppTest.from_file(APP_FILE, default_timeout=30).run()
            _element(fresh.sidebar.text_input, 'Data Directory Path').set_value(temporary).run()
            fresh.button(key='resume_curation').click().run()
            self.assertFalse(fresh.exception)
            self.assertEqual(fresh.session_state['trajectory_dict']['FOV_0_CH_1']['excluded_indices'], {2})

    def test_next_is_only_forward_button_and_review_survives_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            app = _saved_app(directory)
            self.assertEqual(sum(button.label == 'Next ➡️' for button in app.button), 1)
            self.assertFalse(any(button.key == 'mark_reviewed_next' for button in app.button))
            _element(app.button, 'Next ➡️').click().run()
            self.assertFalse(app.exception)
            saved = load_dataset(directory / DB_FILENAME, 1)
            self.assertEqual(saved['trajectory_dict']['FOV_0_CH_1']['_review_status'], 'reviewed')
            self.assertEqual(saved['trajectory_dict']['FOV_0_CH_2']['_review_status'], 'unreviewed')
            self.assertEqual(saved['current_key'], 'FOV_0_CH_2')
            timestamp = saved['trajectory_dict']['FOV_0_CH_1']['_reviewed_at']
            fresh = AppTest.from_file(APP_FILE, default_timeout=30).run()
            _element(fresh.sidebar.text_input, 'Data Directory Path').set_value(temporary).run()
            fresh.button(key='resume_curation').click().run()
            self.assertFalse(fresh.exception)
            _element(fresh.button, '⬅️ Prev').click().run()
            self.assertEqual(fresh.session_state['trajectory_dict']['FOV_0_CH_1']['_review_status'], 'reviewed')
            self.assertEqual(fresh.session_state['trajectory_dict']['FOV_0_CH_1']['_reviewed_at'], timestamp)

    def test_autopilot_merge_and_replace_skip_reviewed_destinations(self):
        from .test_streamlit_division_resolution_panel import _save_tiny_checkpoint, _load_tiny_ai_model

        for policy in ('Merge', 'Replace'):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                app = _saved_app(directory)
                # Confirm both, including last-to-first wrap, before arming AI.
                _element(app.button, 'Next ➡️').click().run()
                _select_plot_points(app, [{'customdata':[2]}])
                _element(app.button, '🚫 Exclude Selected').click().run()
                _element(app.button, 'Next ➡️').click().run()
                before = load_dataset(directory / DB_FILENAME, 1)
                checkpoint = directory / 'tiny_model.pt'
                _save_tiny_checkpoint(str(checkpoint))
                _load_tiny_ai_model(app, str(checkpoint))
                _element(app.radio, 'Apply policy').set_value(policy).run()
                _element(app.toggle, 'Auto pilot on Next').set_value(True).run()
                with patch(
                    'moma_lineage_analysis.trajectory_cleaning.streamlit_api.predict_trajectory_scope',
                    side_effect=AssertionError('Reviewed trajectory must not invoke inference'),
                ):
                    _element(app.button, 'Next ➡️').click().run()
                    self.assertFalse(app.exception)
                    self.assertEqual(app.session_state['current_idx'], 1)
                    self.assertEqual(app.session_state['ai_autopilot_last_status'], 'skipped_reviewed')
                    self.assertEqual(app.session_state['ai_autopilot_completed_count'], 0)
                    self.assertEqual(app.session_state['ai_predictions'], {})
                    _element(app.button, 'Next ➡️').click().run()
                    self.assertFalse(app.exception)
                    self.assertEqual(app.session_state['current_idx'], 0)
                after = load_dataset(directory / DB_FILENAME, 1)
                for key, entry in before['trajectory_dict'].items():
                    restored = after['trajectory_dict'][key]
                    self.assertEqual(set(restored['div_indices']), set(entry['div_indices']))
                    self.assertEqual(restored['excluded_indices'], entry['excluded_indices'])
                    self.assertEqual(restored['_review_status'], 'reviewed')
                    self.assertEqual(restored['_reviewed_at'], entry['_reviewed_at'])
                self.assertIsNone(app.session_state['ai_model_error'])

    def test_autopilot_single_trajectory_next_confirms_without_applying_ai(self):
        from .test_streamlit_division_resolution_panel import _save_tiny_checkpoint, _load_tiny_ai_model

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            entries, _ = _fixture(directory)
            entry = entries['FOV_0_CH_1']
            app = _app_with_entries({'FOV_0_CH_1':entry})
            checkpoint = directory / 'tiny_model.pt'
            _save_tiny_checkpoint(str(checkpoint))
            _load_tiny_ai_model(app, str(checkpoint))
            _element(app.radio, 'Apply policy').set_value('Replace').run()
            _element(app.toggle, 'Auto pilot on Next').set_value(True).run()
            _element(app.button, 'Next ➡️').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['current_idx'], 0)
            self.assertEqual(app.session_state['trajectory_dict']['FOV_0_CH_1']['_review_status'], 'reviewed')
            self.assertEqual(app.session_state['trajectory_dict']['FOV_0_CH_1']['div_indices'], entry['div_indices'])
            self.assertEqual(app.session_state['trajectory_dict']['FOV_0_CH_1']['excluded_indices'], entry['excluded_indices'])
            self.assertEqual(app.session_state['ai_autopilot_completed_count'], 0)
            self.assertEqual(app.session_state['ai_autopilot_last_status'], 'skipped_reviewed')
            self.assertEqual(app.session_state['ai_predictions'], {})

    def test_ai_all_scope_preserves_reviewed_trajectory(self):
        from .test_streamlit_division_resolution_panel import _save_tiny_checkpoint, _load_tiny_ai_model

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            app = _saved_app(directory)
            _element(app.button, 'Next ➡️').click().run()
            before = load_dataset(directory / DB_FILENAME, 1)['trajectory_dict']['FOV_0_CH_1']
            checkpoint = directory / 'tiny_model.pt'
            _save_tiny_checkpoint(str(checkpoint))
            _load_tiny_ai_model(app, str(checkpoint))
            _element(app.selectbox, 'Scope').set_value('All').run()
            _element(app.radio, 'Apply policy').set_value('Replace').run()
            _element(app.button, '✨ Run AI Preview').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(set(app.session_state['ai_predictions']), {'FOV_0_CH_2'})
            _element(app.button, '✅ Apply AI Suggestions').click().run()
            self.assertFalse(app.exception)
            after = load_dataset(directory / DB_FILENAME, 1)['trajectory_dict']['FOV_0_CH_1']
            self.assertEqual(set(after['div_indices']), set(before['div_indices']))
            self.assertEqual(after['excluded_indices'], before['excluded_indices'])
            self.assertEqual(after['_review_status'], 'reviewed')
            self.assertEqual(after['_reviewed_at'], before['_reviewed_at'])

    def test_retry_initial_network_save_preserves_unsaved_session_edits(self):
        from moma_lineage_analysis.curation_store import CurationStoreError

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / 'fov_4_cells_df.csv'
            pd.DataFrame({
                'time_s':np.arange(8) * 3600.0, 'time_point':[f't{i}.tiff' for i in range(8)],
                'channels':1, 'cell_index':1, 'spine_length':np.arange(8) * 0.3 + 2,
            }).to_csv(source, index=False)
            app = AppTest.from_file(APP_FILE, default_timeout=30).run()
            _element(app.sidebar.text_input, 'Data Directory Path').set_value(temporary).run()
            _element(app.sidebar.number_input, 'Min Duration (h)').set_value(0.0).run()
            _element(app.sidebar.slider, 'Number of Cores').set_value(1).run()
            with patch('moma_lineage_analysis.app.curation_session.create_dataset',
                       side_effect=CurationStoreError('create_dataset: invalid uri authority: server')):
                _element(app.sidebar.button, 'Process Data').click().run()
                self.assertFalse(app.exception)
                self.assertIn('invalid uri authority', app.session_state['curation_error'])
                self.assertIn('curation_pending_initial', app.session_state)
                _select_plot_points(app, [{'customdata':[2]}])
                _element(app.button, '🚫 Exclude Selected').click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state['trajectory_dict']['FOV_4_CH_1']['excluded_indices'], {2})
            app.button(key='retry_curation_save').click().run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state['curation_error'])
            self.assertNotIn('curation_pending_initial', app.session_state)
            saved = load_dataset(directory / DB_FILENAME, 1)
            self.assertEqual(saved['trajectory_dict']['FOV_4_CH_1']['excluded_indices'], {2})
            self.assertEqual(len(saved['trajectory_dict']['FOV_4_CH_1']['trajectory']), 8)

    def test_process_data_creates_database_and_reuses_identical_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / 'fov_4_cells_df.csv'
            pd.DataFrame({
                'time_s':np.arange(8) * 3600.0, 'time_point':[f't{i}.tiff' for i in range(8)],
                'channels':1, 'cell_index':1, 'spine_length':np.arange(8) * 0.3 + 2,
            }).to_csv(source, index=False)
            app = AppTest.from_file(APP_FILE, default_timeout=30).run()
            _element(app.sidebar.text_input, 'Data Directory Path').set_value(temporary).run()
            _element(app.sidebar.number_input, 'Min Duration (h)').set_value(0.0).run()
            _element(app.sidebar.slider, 'Number of Cores').set_value(1).run()
            _element(app.sidebar.button, 'Process Data').click().run()
            self.assertFalse(app.exception)
            self.assertTrue((directory / DB_FILENAME).is_file())
            _element(app.button, 'Next ➡️').click().run()
            _element(app.sidebar.button, 'Process Data').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['curation_dataset_id'], 1)
            self.assertEqual(next(iter(app.session_state['trajectory_dict'].values()))['_review_status'], 'reviewed')
