"""Selection invariants for R016: test data excluded, controls separate, seeds fixed."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('temporal_sweep',Path(__file__).resolve().parents[2]/'scripts/sweep_temporal_context.py')
sweep=importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def row(architecture,context,seed,division,exclusion,control=False):
    metrics={'validation':{'division_event':{'f1':division},'point':{'excluded_point':{'f1':exclusion}}},'minimum_validation_loss':.01}
    return dict(architecture=architecture,context_frames=context,seed=seed,control=control,
                run_id=f'{architecture}_{context}_{seed}',selection_score=sweep.validation_score(metrics),parameter_count=100)


def test_selection_rejects_metrics_with_test_block():
    with pytest.raises(ValueError,match='validation-only'):
        sweep.validation_score({'test':{'f1':1.0}})


def test_control_cannot_win_context_shortlist_and_tie_prefers_smaller():
    rows=[row(a,c,42,.9,.8) for a in ('cnn','resnet') for c in (65,17,33,49)]
    rows.append(row('resnet',121,42,1.0,1.0,True))
    assert sweep.shortlist(rows)=={'cnn':[17,33,49],'resnet':[17,33,49]}


def test_repeat_aggregates_all_seeds_instead_of_choosing_best_seed():
    records=[row('cnn',65,s,.9,e) for s,e in [(42,1.0),(43,.6),(44,.5)]]
    aggregate=sweep.aggregate_rows(records)
    assert aggregate['selection_score']['mean_f1']==pytest.approx((.95+.75+.7)/3)
    assert aggregate['sample_sd']['mean_f1']>0
    assert aggregate['seeds']==[42,43,44]
    assert aggregate['runs']==['cnn_65_42','cnn_65_43','cnn_65_44']


def test_tie_breaks_exclusion_before_validation_loss():
    high_exclusion=row('cnn',65,42,.8,1.)
    high_division=row('cnn',17,42,1.,.8)
    assert sweep.rank_key(high_exclusion)<sweep.rank_key(high_division)
