"""Navigation must not mutate an already-instantiated selectbox."""
import importlib.util
from pathlib import Path
from streamlit.testing.v1 import AppTest
from moma_lineage_analysis.paths import PACKAGE_ROOT


def test_next_previous_without_database_mutation(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('store_fixture', Path(__file__).with_name('test_store.py'))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    data = fixture._build_store(tmp_path)
    monkeypatch.setenv('MOMA_TRAINSET_STORE', str(data['db']))
    app = AppTest.from_file(str(PACKAGE_ROOT / 'trainset_review/streamlit_app.py'), default_timeout=30).run()
    pick = lambda kind, label: next(e for e in getattr(app, kind) if e.label == label)
    pick('selectbox', 'Dataset version').set_value('corpus_a@v1').run()
    pick('selectbox', 'Trajectory').set_value('t1').run()
    pick('button', 'Next →').click().run()
    assert not app.exception and not app.error
    assert pick('selectbox', 'Trajectory').value == 't2'
    pick('button', '← Previous').click().run()
    assert not app.exception and not app.error
    assert pick('selectbox', 'Trajectory').value == 't1'
    from moma_lineage_analysis.trainset_review import store
    assert store.revision(data['db']) == 0


def assert_position(app, number, trajectory):
    pick = lambda kind, label: next(e for e in reversed(list(getattr(app, kind))) if e.label == label)
    assert not app.exception and not app.error
    assert pick('selectbox', 'Trajectory').value == trajectory
    assert pick('number_input', 'Trajectory number').value == number
    assert pick('slider', 'Trajectory position').value == number
    # The plot's identity follows the selected trajectory on every control.
    import json
    assert json.loads(app.get('plotly_chart')[-1].proto.spec)['layout']['title']['text'] == trajectory


def review_fixture(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('store_fixture', Path(__file__).with_name('test_store.py'))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    data = fixture._build_store(tmp_path)
    monkeypatch.setenv('MOMA_TRAINSET_STORE', str(data['db']))
    app = AppTest.from_file(str(PACKAGE_ROOT / 'trainset_review/streamlit_app.py'), default_timeout=30).run()
    next(e for e in app.selectbox if e.label == 'Dataset version').set_value('corpus_a@v1').run()
    return app, data


def test_slider_number_dropdown_and_arrows_stay_synchronized(tmp_path, monkeypatch):
    app, data = review_fixture(tmp_path, monkeypatch)
    pick = lambda kind, label: next(e for e in reversed(list(getattr(app, kind))) if e.label == label)
    assert_position(app, 1, 't1')
    pick('slider', 'Trajectory position').set_value(3).run()
    assert_position(app, 3, 't_long')
    pick('number_input', 'Trajectory number').set_value(2).run()
    assert_position(app, 2, 't2')
    pick('selectbox', 'Trajectory').set_value('t1').run()
    assert_position(app, 1, 't1')
    pick('button', 'Next →').click().run()
    assert_position(app, 2, 't2')
    pick('button', '← Previous').click().run()
    assert_position(app, 1, 't1')
    from moma_lineage_analysis.trainset_review import store
    assert store.revision(data['db']) == 0


def test_navigation_resynchronizes_on_filter_sort_and_lineage_changes(tmp_path, monkeypatch):
    app, data = review_fixture(tmp_path, monkeypatch)
    pick = lambda kind, label: next(e for e in reversed(list(getattr(app, kind))) if e.label == label)
    pick('slider', 'Trajectory position').set_value(3).run()
    pick('text_input', 'Find trajectory').set_value('t2').run()
    assert_position(app, 1, 't2')
    assert pick('slider', 'Trajectory position').disabled
    assert pick('number_input', 'Trajectory number').disabled
    pick('text_input', 'Find trajectory').set_value('no matches').run()
    assert not app.exception and not app.error
    assert not any(e.label == 'Trajectory number' for e in app.number_input)
    pick('text_input', 'Find trajectory').set_value('').run()
    pick('selectbox', 'Trajectory').set_value('t2').run()
    pick('selectbox', 'Queue order').set_value('Most exclusions').run()
    assert_position(app, 2, 't2')  # t_long, t2, t1
    pick('number_input', 'Trajectory number').set_value(3).run()
    assert_position(app, 3, 't1')
    pick('selectbox', 'Dataset version').set_value('corpus_b@v1').run()
    assert_position(app, 1, 't1')
    pick('selectbox', 'Trajectory').set_value('t3').run()
    assert_position(app, 2, 't3')
    from moma_lineage_analysis.trainset_review import store
    assert store.revision(data['db']) == 0


def test_reannotation_reuses_figure_and_keeps_preview_cached(tmp_path, monkeypatch):
    app, data = review_fixture(tmp_path, monkeypatch)
    pick = lambda kind, label: next(e for e in reversed(list(getattr(app, kind))) if e.label == label)
    pick('button', 'Run model preview [P]').click().run()
    assert not app.exception
    first = app.session_state['review_figure']
    first_figure = first['figure']
    first_context = first['context']
    first_predictions = app.session_state['review_predictions']['result']['predictions']
    assert 'review_predictions' in app.session_state
    app.session_state['review_selected_ids'] = ['1']
    app.run()
    pick('button', 'Add division [D]').click().run()
    assert not app.exception and not app.error
    second = app.session_state['review_figure']
    assert second['figure'] is first_figure  # Only the traces were replaced.
    assert second['context'] == first_context
    assert any(trace.name == 'Division label' for trace in second['figure'].data)
    assert 'review_predictions' in app.session_state  # Prediction is label-independent.
    # Reusing the same preview must not repeat model inference.
    assert app.session_state['review_predictions']['result']['predictions'] is first_predictions


def test_label_action_does_not_request_an_extra_page_rerun(tmp_path, monkeypatch):
    import streamlit as st
    app, _ = review_fixture(tmp_path, monkeypatch)
    pick = lambda kind, label: next(e for e in reversed(list(getattr(app, kind))) if e.label == label)
    app.session_state['review_selected_ids'] = ['1']
    app.run()
    calls = []
    monkeypatch.setattr(st, 'rerun', lambda *args, **kwargs: calls.append((args, kwargs)))
    pick('button', 'Add division [D]').click().run()
    assert not app.exception and not app.error
    assert calls == []  # Widget callbacks already cause the one required rerun.


def test_removal_resynchronizes_position_controls(tmp_path, monkeypatch):
    app, data = review_fixture(tmp_path, monkeypatch)
    pick = lambda kind, label: next(e for e in reversed(list(getattr(app, kind))) if e.label == label)
    pick('slider', 'Trajectory position').set_value(2).run()
    pick('button', 'Remove trajectory from database [Delete]').click().run()
    assert_position(app, 2, 't_long')
    pick('button', 'Remove trajectory from database [Delete]').click().run()
    assert_position(app, 1, 't1')
    pick('button', 'Remove trajectory from database [Delete]').click().run()
    assert not app.exception and not app.error
    assert not any(e.label == 'Trajectory number' for e in app.number_input)
    assert any('No trajectories match' in e.value for e in app.info)