"""Shortcut labels and iframe bindings stay aligned with real dataset actions."""
import importlib.util
import json
from pathlib import Path
import re

import pytest
from moma_lineage_analysis.trainset_review import store

spec = importlib.util.spec_from_file_location('review_app_tests', Path(__file__).with_name('test_streamlit_app.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def bindings(app):
    script = app.get('iframe')[-1].proto.srcdoc
    return json.loads(re.search(r'const bindings = (.*);', script).group(1))


def test_every_mutation_button_has_matching_visible_shortcut(tmp_path, monkeypatch):
    fixture = helpers.fixture_store(tmp_path)
    app = helpers.app_for(monkeypatch, fixture['db'])
    expected = {
        'd': 'Add division [D]', 'm': 'Remove division [M]',
        'e': 'Exclude points [E]', 'r': 'Restore points [R]', 'c': 'Clear labels [C]',
        'f': 'Resolve division/exclusion conflicts [F]', 'u': 'Undo last label edit [U]',
        'k': 'Mark reviewed [K]', 'Delete': 'Remove trajectory from database [Delete]',
        't': 'Make training-ready [T]',
    }
    actual = bindings(app)
    for key, label in expected.items():
        assert actual[key] == label
        helpers.element(app.button, label)
    helpers.element(app.button, 'Mark reviewed [K]').click().run()
    assert bindings(app)['k'] == 'Mark not reviewed [K]'
    helpers.element(app.button, 'Run model preview [P]').click().run()
    assert not app.exception
    assert bindings(app)['a'] == 'Apply selected suggestion types [A]'
    helpers.element(app.checkbox, 'Apply division predictions').uncheck().run()
    helpers.element(app.checkbox, 'Apply exclusion predictions').uncheck().run()
    assert helpers.element(app.button, 'Apply selected suggestion types [A]').disabled


@pytest.mark.parametrize('label,selected,expected_division,expected_excluded', [
    ('Add division [D]', '1', True, False),
    ('Remove division [M]', '2', False, False),
    ('Exclude points [E]', '1', False, True),
    ('Restore points [R]', '1', False, False),
    ('Clear labels [C]', '2', False, False),
])
def test_labeled_actions_autosave_expected_labels(tmp_path, monkeypatch, label, selected,
                                                expected_division, expected_excluded):
    fixture = helpers.fixture_store(tmp_path)
    db = fixture['db']
    app = helpers.app_for(monkeypatch, db)
    helpers.element(app.selectbox, 'Dataset version').set_value('corpus_a@v1').run()
    helpers.element(app.selectbox, 'Trajectory').set_value('t1').run()
    if label == 'Restore points [R]':
        helpers.select_ids(app, [selected])
        helpers.element(app.button, 'Exclude points [E]').click().run()
    helpers.select_ids(app, [selected])
    helpers.element(app.button, label).click().run()
    assert not app.exception and not app.error
    row = store.load_trajectory(db, 't1').set_index('fov_cell_index').loc[int(selected)]
    assert bool(row.division_point) == expected_division
    assert bool(row.excluded_point) == expected_excluded
