"""R022 fixed-split exclusion study: prepare, train (validation only), then lock.

The separate exclusion_compare.py command may evaluate test data only after a
validation-only selection lock. No command modifies the live dataset or model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
import torch
from moma_lineage_analysis.trajectory_cleaning import dataset_store as ds
from moma_lineage_analysis.trajectory_cleaning.data import (
    TrajectoryDataset, TrajectoryNormalizer, split_trajectory_keys,
)
from moma_lineage_analysis.trajectory_cleaning.models import ModelConfig, build_model
from moma_lineage_analysis.trajectory_cleaning.training import (
    TrainConfig, collect_predictions, evaluate_model, save_checkpoint,
    set_reproducible_seed, train_model,
)

OUT = ROOT / 'runs/exclusion_20261110'
PRODUCTION = ROOT / 'src/moma_lineage_analysis/models/trajectory_cleaning_run'
PRODUCTION_SHA = '816a8b392ec206dca578a093c6cbc2fc9ec5ba12480d6aca7c8b08349796af99'
LINEAGES = [
    'reviewed_20261031_verified_20260916_no_longruns@v1',
    'reviewed_20261031_verified_20261013@v1',
    'reviewed_20261031_verified_20261016@v1',
    'reviewed_20261031_verified_20261017@v1',
    'reviewed_20261031_verified_20261029@v1',
    'verified_20261109@v1',
]
ARMS = {
    'baseline': {},
    'cap100': {'maximum_positive_weight': 100.0},
    'cap222': {'maximum_positive_weight': 222.0},
    'exclusion_x2': {'exclusion_loss_weight': 2.0},
    'exclusion_x3': {'exclusion_loss_weight': 3.0},
    'exclusion_windows': {'sampling': 'exclusion_windows'},
    'exclusion_ap': {'checkpoint_metric': 'exclusion_ap'},
}
SEEDS = [42, 43, 44]
SOURCE_PATHS = [
    'scripts/exclusion_study.py',
    'src/moma_lineage_analysis/trajectory_cleaning/training.py',
    'src/moma_lineage_analysis/trajectory_cleaning/sampling.py',
    'src/moma_lineage_analysis/trajectory_cleaning/data.py',
    'src/moma_lineage_analysis/trajectory_cleaning/models.py',
]


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def signatures(frame):
    """Keys with the same immutable signal must never straddle partitions."""
    groups = {}
    for key, group in frame.groupby('trajectory_key', sort=False):
        group = group.sort_values(['time_h', 'fov_cell_index'], kind='stable')
        # Ignore IDs: identical time/length signals remain twins under renamed IDs.
        pairs = np.column_stack((group.time_h.to_numpy('float64'),
                                 group.spine_length.to_numpy('float64')))
        signature = hashlib.sha256(pairs.astype('<f8').tobytes()).hexdigest()
        groups.setdefault(signature, []).append(str(key))
    return groups


def summarize(frame, splits):
    result = {}
    for partition, keys in splits.items():
        f = frame[frame.trajectory_key.isin(keys)]
        per = f.groupby('trajectory_key').excluded_point.sum()
        result[partition] = dict(rows=len(f), trajectories=len(keys),
            divisions=int(f.division_point.sum()), exclusions=int(f.excluded_point.sum()),
            exclusion_positive_trajectories=int((per > 0).sum()),
            by_source={str(k): int(v) for k,v in
                       f.groupby('source_dataset').trajectory_key.nunique().items()})
    return result


def prepare(output=OUT):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        raise ValueError('Protocol exists; do not replace a registered study')
    if ds.sha256_file(PRODUCTION / 'best_model.pt') != PRODUCTION_SHA:
        raise ValueError('Production checkpoint changed')
    connection = ds.connect_readonly(ROOT / 'datasets/store/lineage.sqlite3')
    try:
        frame, lineage = ds.load_training_frame(connection, LINEAGES,
            store_path=ROOT / 'datasets/store/lineage.sqlite3', require_human_verified=True)
        revision = int(connection.execute('SELECT revision FROM review_state').fetchone()[0])
    finally:
        connection.close()
    original = read_json(PRODUCTION / 'splits.json')
    original_assignment = {key:p for p,keys in original.items() for key in keys}
    new = frame[frame.source_dataset.eq('curated_20261109')]
    # New source is split without changing the historical assignments.
    added = split_trajectory_keys(new, seed=42, source_column='source_dataset')
    splits = {p:list(keys) + list(added[p]) for p,keys in original.items()}
    assignment = {key:p for p,keys in splits.items() for key in keys}
    if set(assignment) != set(frame.trajectory_key.unique()):
        raise ValueError('Unexpected population relative to production plus new batch')
    priority = {'train':0, 'validation':1, 'test':2}
    duplicates = []
    omitted = set()
    for signature, keys in signatures(frame).items():
        if len(keys) < 2:
            continue
        representative = min(keys, key=lambda k: (priority[assignment[k]], k))
        removed = sorted(set(keys) - {representative})
        omitted.update(removed)
        duplicates.append(dict(signal_sha256=signature, retained=representative,
            retained_partition=assignment[representative], omitted=removed,
            original_partitions={k:assignment[k] for k in keys},
            reason='retain one immutable signal; prefer train then validation then test'))
    splits = {p:[k for k in keys if k not in omitted] for p,keys in splits.items()}
    frame = frame[~frame.trajectory_key.isin(omitted)].reset_index(drop=True)
    all_keys = [k for keys in splits.values() for k in keys]
    if len(set(all_keys)) != len(all_keys):
        raise ValueError('Key leakage')
    assignment = {k:p for p,keys in splits.items() for k in keys}
    for keys in signatures(frame).values():
        if len({assignment[k] for k in keys}) != 1:
            raise ValueError('Signal-twin leakage')
    snapshot = output / 'frame.pkl'
    frame.to_pickle(snapshot)  # private, generated study snapshot; load only after SHA check
    write_json(output / 'splits.json', splits)
    write_json(output / 'dataset_lineage.json', lineage)
    write_json(output / 'duplicate_exclusions.json', duplicates)
    write_json(output / 'split_summary.json', summarize(frame, splits))
    # Input hashes protect the deployed release as well as the source population.
    preserved = {str(p.relative_to(ROOT)):ds.sha256_file(p)
                 for p in (PRODUCTION / n for n in
                           ['best_model.pt','history.csv','metrics.json','splits.json','promotion.json'])}
    preserved['datasets/manifest.json'] = ds.sha256_file(ROOT / 'datasets/manifest.json')
    write_json(output / 'preservation_before.json', preserved)
    protocol = dict(round='R022',registered_at=now(),dataset_lineages=LINEAGES,
        dataset_revision=revision,dataset_snapshot='frame.pkl',
        dataset_snapshot_sha256=ds.sha256_file(snapshot),
        split_sha256=ds.sha256_file(output/'splits.json'),
        production_sha256=PRODUCTION_SHA,production_splits_sha256=ds.sha256_file(PRODUCTION/'splits.json'),
        population=dict(rows=len(frame),trajectories=len(all_keys),
                        exclusions=int(frame.excluded_point.sum()),divisions=int(frame.division_point.sum())),
        split_rule='Preserve R019 assignments; source-stratified seed42 split new batch; '
                   'remove redundant exact time/length twins preferring train > validation > test.',
        duplicate_exclusions=duplicates,seeds=SEEDS,arms=ARMS,
        model=ModelConfig(model_type='cnn').to_dict(),
        training_base=dict(epochs=100,batch_size=16,learning_rate=0.001,weight_decay=0.0001,
            focal_gamma=1.5,maximum_positive_weight=50.0,patience=15,
            exclusion_loss_weight=1.0,checkpoint_metric='validation_loss',sampling='trajectory',
            window_core=65,window_positive_fraction=0.5,division_tolerance_frames=1),
        sampling_policy='Training records only; normalize full trajectory then crop core65 plus RF8 '
          'halo each side; half event-centered and half uniform-row centers; samples per epoch '
          '= training trajectories; original-train pos weights; full-trajectory validation.',
        selection=dict(primary='mean validation exclusion AP across seeds42/43/44',
            guardrails='mean validation division-event F1 >= baseline mean -0.002; '
                       'mean validation exclusion F1 >= baseline mean -0.01',
            tie_break=['mean exclusion F1','mean division-event F1','registered arm order'],
            seed_selection='none; compare all seeds, fixed seed42 as reproducible representative',
            fallback='baseline remains eligible; choose it if no improved arm qualifies',
            threshold='existing validation-only point-F1 grid0.05..0.95 per label'),
        evaluation=dict(test_inference_before_selection_lock=False,
            primary='Deduplicated R019 test trajectories plus fixed new-batch test trajectories; '
                    'exclude any immutable-signal twin of production training.',
            controls=['R019 production deployed thresholds','R019 production thresholds recalibrated '
                      'on R022 validation only','R022 baseline seeds42/43/44'],
            panels=['all clean test','historical clean test','new-batch clean test'],
            uncertainty='Paired trajectory-cluster bootstrap,5000 replicates,seed20261110; '
                        'AP/F1/recall/precision and division-event F1; seed SD reported separately',
            no_promotion=True,
            limitations=['reused historical validation/test; not a fresh confirmatory sample',
                'trajectory-grouped, not FOV-held-out; shared acquisition context may leak',
                'few new-batch exclusion positives; three seeds measure optimization noise',
                'window arm has equal updates but fewer loss points per update',
                'changing class weights need not improve ranking; tests cannot select models']))
    write_json(output/'protocol.json',protocol)
    print(json.dumps({'prepared':protocol['population'], 'split':summarize(frame,splits),
                      'duplicate_exclusions':duplicates},indent=2))


def load_study(output=OUT):
    protocol = read_json(output/'protocol.json')
    for name,expected in [('frame.pkl',protocol['dataset_snapshot_sha256']),
                          ('splits.json',protocol['split_sha256'])]:
        if ds.sha256_file(output/name) != expected:
            raise ValueError(f'Frozen {name} changed')
    if ds.sha256_file(PRODUCTION/'best_model.pt') != protocol['production_sha256']:
        raise ValueError('Deployed checkpoint changed')
    frame=pd.read_pickle(output/'frame.pkl')
    return protocol,frame,read_json(output/'splits.json')


def train_one(arm,seed,output=OUT,device='cuda:0'):
    protocol,frame,splits=load_study(output)
    if arm not in protocol['arms'] or seed not in protocol['seeds']:
        raise ValueError('Unregistered run')
    run=output/'training'/arm/f'seed{seed}'
    if (output/'selection_lock.json').exists():
        raise ValueError('Training closed after validation selection lock')
    if (run/'validation.json').exists():
        old=read_json(run/'validation.json')
        if old['protocol_sha256'] != ds.sha256_file(output/'protocol.json'):
            raise ValueError('Existing run protocol mismatch')
        if ds.sha256_file(run/'best_model.pt') != old['checkpoint_sha256']:
            raise ValueError('Completed checkpoint changed')
        if old['source_hashes'] != {p:ds.sha256_file(ROOT/p) for p in SOURCE_PATHS}:
            raise ValueError('Training code changed since completed run')
        print(f'SKIP completed {arm} seed{seed}',flush=True)
        return
    run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    normalizer=TrajectoryNormalizer()
    train=TrajectoryDataset(frame,keys=splits['train'],normalizer=normalizer)
    validation=TrajectoryDataset(frame,keys=splits['validation'],normalizer=normalizer)
    parameters={**protocol['training_base'],**protocol['arms'][arm], 'seed':seed,'device':device}
    config=TrainConfig(**parameters)
    model_config=ModelConfig(**protocol['model'])
    set_reproducible_seed(seed)
    model=build_model(model_config)
    source_hashes={p:ds.sha256_file(ROOT/p) for p in SOURCE_PATHS}
    start=time.monotonic()
    print(f'START {arm} seed{seed} {now()}',flush=True)
    model,history,thresholds=train_model(model,train,validation,config=config)
    validation_metrics=evaluate_model(model,validation,thresholds=thresholds,
        device=device,batch_size=16,division_tolerance_frames=1)
    metadata=dict(study_round='R022',arm=arm,seed=seed,train_config=config.to_dict(),
        source_hashes=source_hashes,protocol_sha256=ds.sha256_file(output/'protocol.json'),
        splits_sha256=protocol['split_sha256'],dataset_lineage=read_json(output/'dataset_lineage.json'),
        test_evaluated=False,trajectory_column='trajectory_key',time_column='time_h',length_column='spine_length')
    save_checkpoint(run/'best_model.pt',model,model_config=model_config,
                    normalizer=normalizer,thresholds=thresholds,metadata=metadata)
    pd.DataFrame(history).to_csv(run/'history.csv',index=False)
    write_json(run/'splits.json',splits)
    records=collect_predictions(model,validation,batch_size=16,device=device)
    # Numeric archive with explicit string keys; never object arrays/pickle.
    np.savez_compressed(run/'validation_predictions.npz',
        keys=np.array([str(r['key']) for r in records]),
        lengths=np.array([len(r['targets']) for r in records]),
        probabilities=np.concatenate([r['probabilities'] for r in records]),
        targets=np.concatenate([r['targets'] for r in records]))
    out=dict(arm=arm,seed=seed,validation=validation_metrics,
        thresholds=[float(x) for x in thresholds],epochs=len(history),seconds=time.monotonic()-start,
        train_config=config.to_dict(),model_config=model_config.to_dict(),
        protocol_sha256=metadata['protocol_sha256'],source_hashes=source_hashes,
        checkpoint_sha256=ds.sha256_file(run/'best_model.pt'),test_evaluated=False,
        device_name=torch.cuda.get_device_name(torch.device(device)) if device.startswith('cuda') else 'cpu')
    write_json(run/'validation.json',out)
    print(f"DONE {arm} seed{seed}: AP={validation_metrics['point']['excluded_point']['average_precision']:.6f} "
          f"F1={validation_metrics['point']['excluded_point']['f1']:.6f} "
          f"divF1={validation_metrics['division_event']['f1']:.6f}; {len(history)} epochs; {out['seconds']:.1f}s",flush=True)


def lock(output=OUT):
    protocol=read_json(output/'protocol.json')
    if (output/'selection_lock.json').exists():
        raise ValueError('Selection already locked')
    if (output/'evaluation').exists():
        raise ValueError('Evaluation artifacts exist before selection')
    rows=[]
    source_reference=None
    validation_hashes={}
    for arm in protocol['arms']:
        runs=[read_json(output/'training'/arm/f'seed{s}'/'validation.json') for s in protocol['seeds']]
        if any(r.get('test_evaluated') is not False for r in runs):
            raise ValueError('Test evaluated during tuning')
        if any(r['protocol_sha256'] != ds.sha256_file(output/'protocol.json') for r in runs):
            raise ValueError('Protocol mismatch')
        for r in runs:
            run=output/'training'/arm/f"seed{r['seed']}"
            if ds.sha256_file(run/'best_model.pt') != r['checkpoint_sha256']:
                raise ValueError('Checkpoint changed before lock')
            if source_reference is None:
                source_reference=r['source_hashes']
            if r['source_hashes'] != source_reference:
                raise ValueError('Training code differs between runs')
            validation_hashes[f"{arm}/seed{r['seed']}"]=ds.sha256_file(run/'validation.json')
        values={
            'exclusion_ap':[r['validation']['point']['excluded_point']['average_precision'] for r in runs],
            'exclusion_f1':[r['validation']['point']['excluded_point']['f1'] for r in runs],
            'division_event_f1':[r['validation']['division_event']['f1'] for r in runs]}
        rows.append(dict(arm=arm,seeds=protocol['seeds'],mean={k:statistics.mean(v) for k,v in values.items()},
            seed_sd={k:statistics.stdev(v) for k,v in values.items()},
            checkpoint_sha256={str(r['seed']):r['checkpoint_sha256'] for r in runs}))
    base=next(r for r in rows if r['arm']=='baseline')['mean']
    order=list(protocol['arms'])
    for r in rows:
        r['eligible']=r['mean']['division_event_f1'] >= base['division_event_f1']-0.002 and                       r['mean']['exclusion_f1'] >= base['exclusion_f1']-0.01
    ranked=sorted((r for r in rows if r['eligible']),key=lambda r:(
        -r['mean']['exclusion_ap'],-r['mean']['exclusion_f1'],
        -r['mean']['division_event_f1'],order.index(r['arm'])))
    lock_record=dict(locked_at=now(),protocol_sha256=ds.sha256_file(output/'protocol.json'),
        selected_arm=ranked[0]['arm'],representative_seed=42,seeds=protocol['seeds'],
        selection_uses='validation only',test_evaluated_before_lock=False,
        rows=rows,ranked_eligible=[r['arm'] for r in ranked],production_unchanged=True,
        source_hashes=source_reference,validation_json_sha256=validation_hashes)
    write_json(output/'selection_lock.json',lock_record)
    print(json.dumps(lock_record,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train','train-all','lock'])
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--arm',choices=list(ARMS))
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.output)
    elif args.command=='train':
        if args.arm is None or args.seed is None: parser.error('--arm and --seed required')
        train_one(args.arm,args.seed,args.output,args.device)
    elif args.command=='train-all':
        for arm in ARMS:
            for seed in SEEDS: train_one(arm,seed,args.output,args.device)
    elif args.command=='lock':
        lock(args.output)


if __name__=='__main__':
    main()
