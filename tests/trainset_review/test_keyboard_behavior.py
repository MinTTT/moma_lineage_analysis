"""Behavioral JS checks for the actual shared handler, including Delete safety."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from moma_lineage_analysis.trajectory_view_style import shortcut_script


def test_generated_handler_executes_each_key_and_guards_ineligible_targets(tmp_path):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js is required for keyboard handler behavior checks')
    labels = {
        'ArrowLeft': '← Previous', 'ArrowRight': 'Next →', 'd': 'Add division [D]',
        'm': 'Remove division [M]', 'e': 'Exclude points [E]', 'r': 'Restore points [R]',
        'c': 'Clear labels [C]', 'f': 'Resolve division/exclusion conflicts [F]',
        'u': 'Undo last label edit [U]', 'k': 'Mark reviewed [K]',
        'Delete': 'Remove trajectory from database [Delete]',
        'a': 'Apply selected suggestion types [A]', 't': 'Make training-ready [T]',
        'p': 'Run model preview [P]', 'v': 'Evaluate filtered trajectories [V]',
    }
    payload = tmp_path / 'handler.json'
    payload.write_text(json.dumps({'script':shortcut_script(labels), 'labels':labels}), encoding='utf-8')
    result = subprocess.run([node, str(Path(__file__).with_name('shortcut_behavior.cjs')), str(payload)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['status'] == 'PASS' and report['checks'] >= 50
