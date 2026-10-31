"""CPU inference parity, input validation, review scores, and cache identity."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from moma_lineage_analysis.trainset_review.evaluation import (
    cache_key, checkpoint_sha256, evaluate_frame, split_membership, trajectory_scores,
)
from moma_lineage_analysis.trajectory_cleaning.data import TrajectoryDataset, TrajectoryNormalizer
from moma_lineage_analysis.trajectory_cleaning.models import ModelConfig, build_model
from moma_lineage_analysis.trajectory_cleaning.training import evaluate_model, load_checkpoint, save_checkpoint

BIG = 2**53+11


def frame():
    result = pd.DataFrame({"trajectory_key": ["a"]*4+["b"]*3,
        "fov_cell_index": pd.Series([BIG, BIG+1, BIG+2, BIG+3, BIG, BIG+1, BIG+2], dtype="int64"),
        "time_h": [0.,1.,2.,3.,0.,1.,2.], "spine_length": [3.,4.,2.,3.,4.,2.,3.],
        "division_point": [False,True,False,False,True,False,False],
        "excluded_point": [False,False,True,False,False,False,False]})
    result = result.iloc[[5,2,0,6,4,1,3]].copy(); result.index = [9,9,7,8,4,2,1]
    return result


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(12)
    config = ModelConfig(model_type="cnn", hidden_size=4, num_layers=1, kernel_size=3, dropout=0)
    model = build_model(config)
    with torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
    path = tmp_path/"best_model.pt"
    save_checkpoint(path, model, model_config=config, normalizer=TrajectoryNormalizer(), thresholds=[.4,.6])
    return path


def test_evaluate_uses_stored_thresholds_and_preserves_input_order_and_ids(checkpoint):
    data = frame(); before = data.copy(deep=True)
    result = evaluate_frame(data, checkpoint)
    assert set(result) == {"predictions", "metrics", "thresholds", "checkpoint_sha256"}
    assert result["thresholds"] == np.asarray([.4,.6],dtype=np.float32).tolist()
    assert result["checkpoint_sha256"] == checkpoint_sha256(checkpoint)
    predicted = result["predictions"]
    pd.testing.assert_frame_equal(predicted[before.columns], before)
    pd.testing.assert_frame_equal(data, before)
    np.testing.assert_array_equal(predicted.pred_division_probability, .5)
    assert predicted.pred_division_point.all() and not predicted.pred_excluded_point.any()
    model, normalizer, thresholds, _ = load_checkpoint(checkpoint)
    expected = evaluate_model(model, TrajectoryDataset(data, normalizer=normalizer), thresholds=thresholds,
                              device="cpu", division_tolerance_frames=1)
    assert result["metrics"]["point"] == expected["point"]
    assert result["metrics"]["division_event"] == expected["division_event"]


def test_evaluate_auto_resolves_to_cpu_for_every_inference_stage(checkpoint, monkeypatch):
    from moma_lineage_analysis.trajectory_cleaning import training

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    observed = []
    for name in ("load_checkpoint", "predict_dataframe", "evaluate_model"):
        original = getattr(training, name)

        def checked(*args, _name=name, _original=original, **kwargs):
            assert isinstance(kwargs["device"], torch.device)
            assert kwargs["device"].type == "cpu"
            observed.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(training, name, checked)
    result = evaluate_frame(frame(), checkpoint, device="auto")
    assert observed == ["load_checkpoint", "predict_dataframe", "evaluate_model"]
    assert len(result["predictions"]) == 7
    assert result["predictions"].pred_division_point.all()
    assert not result["predictions"].pred_excluded_point.any()


def test_evaluate_threshold_override_changes_predictions_consistently(checkpoint):
    result = evaluate_frame(frame(), checkpoint, thresholds=[.8,.2])
    assert not result["predictions"].pred_division_point.any()
    assert result["predictions"].pred_excluded_point.all()
    assert result["metrics"]["point"]["division_point"]["threshold"] == np.float32(.8)
    assert result["metrics"]["point"]["excluded_point"]["threshold"] == np.float32(.2)


@pytest.mark.parametrize("thresholds", [[.5], [.5,.5,.5], [np.nan,.5], [.5,np.inf], [-.1,.5], [1.1,.5], [True,.5], [".5",.5]])
def test_bad_thresholds_rejected_before_checkpoint_load(thresholds, tmp_path):
    with pytest.raises(ValueError, match="threshold|Threshold"):
        evaluate_frame(frame(), tmp_path/"missing.pt", thresholds=thresholds)


@pytest.mark.parametrize("column,value", [("time_h",np.nan),("time_h",np.inf),("spine_length",0),
    ("spine_length",-1),("division_point",1),("excluded_point","False"),("fov_cell_index",1.0)])
def test_invalid_source_rejected_before_inference(column, value, tmp_path):
    data = frame(); data[column] = data[column].astype(object); data.iloc[0,data.columns.get_loc(column)] = value
    with pytest.raises(ValueError): evaluate_frame(data, tmp_path/"missing.pt")


def test_duplicate_ids_rejected_within_trajectory_but_not_across(checkpoint):
    data=frame(); positions=np.flatnonzero(data.trajectory_key.eq("a"));data.iloc[positions[1],data.columns.get_loc("fov_cell_index")]=int(data.fov_cell_index.iloc[positions[0]])
    with pytest.raises(ValueError, match="Duplicate"): evaluate_frame(data, checkpoint)
    assert len(evaluate_frame(frame(),checkpoint)["predictions"])==7


def test_trajectory_scores_recompute_flags_and_support_not_stale_prediction_bools():
    data=frame();data["pred_division_probability"]=[.9,.2,.1,.3,.8,.6,.4]
    data["pred_excluded_probability"]=[.1,.8,.2,.3,.7,.4,.1]
    data["pred_division_point"]=False;data["pred_excluded_point"]=False
    before=data.copy(deep=True);scores=trajectory_scores(data,[.5,.5]).set_index("trajectory_key")
    for key,g in data.groupby("trajectory_key",sort=False):
        probs=g[["pred_division_probability","pred_excluded_probability"]].to_numpy()
        target=g[["division_point","excluded_point"]].to_numpy(bool);flag=probs>=.5
        assert scores.loc[key,"division_disagreements"]==int((target[:,0]!=flag[:,0]).sum())
        assert scores.loc[key,"exclusion_disagreements"]==int((target[:,1]!=flag[:,1]).sum())
        assert scores.loc[key,"division_support"]==int(target[:,0].sum())
        assert scores.loc[key,"exclusion_support"]==int(target[:,1].sum())
        assert scores.loc[key,"mean_uncertainty"]==pytest.approx(np.mean(2*np.minimum(probs,1-probs)))
    assert np.isnan(scores.loc["b","exclusion_average_precision"])
    pd.testing.assert_frame_equal(data,before)


@pytest.mark.parametrize("value", [np.nan,np.inf,-.01,1.01])
def test_trajectory_scores_reject_invalid_probability(value):
    data=frame().assign(pred_division_probability=.5,pred_excluded_probability=.5)
    data.iloc[0,data.columns.get_loc("pred_excluded_probability")]=value
    with pytest.raises(ValueError, match="probabilit"): trajectory_scores(data,[.5,.5])


def test_split_membership_reads_companion_and_rejects_overlap(checkpoint):
    path=checkpoint.with_name("splits.json")
    path.write_text(json.dumps({"train":["a"],"validation":["b"],"test":["c"]}),encoding="utf-8")
    assert split_membership(checkpoint,["a","b","c","d"])=={"a":"train","b":"validation","c":"test","d":"unseen"}
    path.write_text(json.dumps({"train":["a"],"validation":["a"],"test":[]}),encoding="utf-8")
    with pytest.raises(ValueError,match="overlap"):split_membership(checkpoint,["a"])
    assert split_membership(checkpoint.parent/"other"/"best_model.pt",["x"])=={"x":"unseen"}


def test_cache_identity_tracks_labels_measurements_order_checkpoint_device_thresholds(checkpoint):
    data=frame();initial=cache_key(data,checkpoint)
    assert initial==cache_key(data.copy(deep=True),checkpoint)
    changed=data.copy();changed.iloc[0,changed.columns.get_loc("division_point")]=not changed.division_point.iloc[0]
    assert cache_key(changed,checkpoint)!=initial
    changed=data.copy();changed.iloc[0,changed.columns.get_loc("spine_length")]+=1
    assert cache_key(changed,checkpoint)!=initial
    assert cache_key(data.iloc[::-1],checkpoint)!=initial
    assert cache_key(data,checkpoint,device="cuda")!=initial  # identity only: no CUDA call
    assert cache_key(data,checkpoint,thresholds=[.5,.5])!=initial
    checkpoint.write_bytes(checkpoint.read_bytes()+b"changed")
    assert cache_key(data,checkpoint)!=initial
