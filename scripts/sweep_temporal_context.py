"""R016 fixed-capacity temporal-context sweep with a validation-only selection stage.

Screen: 16 context candidates + original ResNet control, seed42.
Repeat: best three contexts per architecture, seeds43/44, frozen R015 split.
Evaluate: only after the context choices are locked; report all three winner seeds.
The theoretical context is measured in normalized feature frames, not raw-data
availability: the unchanged normalizer uses whole-trajectory statistics.

Historical note (R016 closed 2026-10-29): the production guard below pins the
R010 CNN checkpoint that was deployed when this experiment ran. R020 promoted
the R019 CNN on 2026-10-31, so the guard now fails against the live package.
To reproduce R016, restore the R010 checkpoint from
`src/moma_lineage_analysis/models/trajectory_cleaning_run/promotion_backups/20261030T184217.243155Z/`
first; do not update the pin, because R016 results refer to that checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from moma_lineage_analysis.trajectory_cleaning import dataset_store as ds
from moma_lineage_analysis.trajectory_cleaning.data import TrajectoryDataset, TrajectoryNormalizer, split_trajectory_keys
from moma_lineage_analysis.trajectory_cleaning.models import ModelConfig, build_model
from moma_lineage_analysis.trajectory_cleaning.training import (
    TrainConfig, _point_metrics, collect_predictions, evaluate_model, load_checkpoint,
    save_checkpoint, set_reproducible_seed, train_model,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'runs/context_20261029'
LABELS = ('division_point', 'excluded_point')


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validation_score(metrics: dict[str, Any]) -> dict[str, float]:
    if 'test' in metrics:
        raise ValueError('Sweep ranking accepts validation-only records')
    value = metrics['validation']
    division = float(value['division_event']['f1'])
    exclusion = float(value['point']['excluded_point']['f1'])
    return {'mean_f1': (division + exclusion) / 2, 'exclusion_f1': exclusion,
            'division_event_f1': division, 'minimum_validation_loss': float(metrics['minimum_validation_loss'])}


def rank_key(row: dict[str, Any]) -> tuple:
    value = row['selection_score']
    return (-value['mean_f1'], -value['exclusion_f1'], -value['division_event_f1'],
            value['minimum_validation_loss'], row['context_frames'])


def shortlist(rows: list[dict[str, Any]], count: int = 3) -> dict[str, list[int]]:
    return {name: [r['context_frames'] for r in sorted(
        [r for r in rows if r['architecture'] == name and not r['control']], key=rank_key)[:count]]
        for name in ('cnn', 'resnet')}


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [r['selection_score'] for r in rows]
    means = {k: statistics.mean(s[k] for s in scores) for k in scores[0]}
    stds = {k: statistics.stdev(s[k] for s in scores) if len(scores) > 1 else 0.0 for k in scores[0]}
    return {'architecture': rows[0]['architecture'], 'context_frames': rows[0]['context_frames'],
            'selection_score': means, 'sample_sd': stds, 'seeds': [r['seed'] for r in rows],
            'runs': [r['run_id'] for r in rows], 'parameter_count': rows[0]['parameter_count']}


class Sweep:
    def __init__(self, output: Path) -> None:
        self.output = output.resolve()
        self.protocol_path = self.output / 'protocol.json'
        self.protocol = read_json(self.protocol_path)
        self.protocol_hash = ds.sha256_file(self.protocol_path)
        conn = ds.connect_readonly(ROOT / self.protocol['dataset']['store'])
        try:
            self.frame, self.lineage = ds.load_training_frame(
                conn, [self.protocol['dataset']['lineage']], require_human_verified=True,
                store_path=ROOT / self.protocol['dataset']['store'])
        finally:
            conn.close()
        if self.lineage['lineages'][0]['canonical_hash'] != self.protocol['dataset']['canonical_hash']:
            raise ValueError('Frozen corpus content changed')
        self.splits_path = ROOT / self.protocol['dataset']['frozen_splits']
        self.splits_hash = ds.sha256_file(self.splits_path)
        self.splits = read_json(self.splits_path)
        canonical = split_trajectory_keys(self.frame, seed=42, source_column='source_dataset')
        if canonical != self.splits:
            raise ValueError('Canonical split differs from frozen R015 split')
        self.normalizer = TrajectoryNormalizer()
        self.train = TrajectoryDataset(self.frame, keys=self.splits['train'], normalizer=self.normalizer)
        self.validation = TrajectoryDataset(self.frame, keys=self.splits['validation'], normalizer=self.normalizer)
        self.source_hashes = {path: ds.sha256_file(ROOT / path) for path in (
            'scripts/sweep_temporal_context.py',
            'src/moma_lineage_analysis/trajectory_cleaning/models.py',
            'src/moma_lineage_analysis/trajectory_cleaning/training.py',
            'src/moma_lineage_analysis/trajectory_cleaning/data.py')}
        baseline = ROOT / 'src/moma_lineage_analysis/models/trajectory_cleaning_run/best_model.pt'
        self.production_hash = ds.sha256_file(baseline)
        if self.production_hash != '6e7c5a5896c90bd0de2b068af087f2d9c942b2de94e8deb87e419fd005cb230b':
            raise ValueError('Production control changed')
        self.dilation_map = dict(zip(self.protocol['architecture']['contexts'], self.protocol['architecture']['dilations']))

    def configuration(self, name: str, context: int, control: bool = False) -> ModelConfig:
        config = ModelConfig(model_type=name, hidden_size=64, num_layers=4,
                             kernel_size=5 if name == 'cnn' or control else 3,
                             conv_dilations=None if control else tuple(self.dilation_map[context]))
        if config.receptive_field != context or not config.has_dense_temporal_support:
            raise ValueError(f'Invalid temporal support: {name}/{context}')
        return config

    def run_path(self, name: str, context: int, seed: int, control: bool = False) -> Path:
        family = 'resnet_original_control' if control else name
        return self.output / family / f'context_{context:03d}' / f'seed_{seed}'

    def run(self, name: str, context: int, seed: int, control: bool = False) -> dict[str, Any]:
        directory = self.run_path(name, context, seed, control)
        result_path = directory / 'validation_metrics.json'
        if result_path.exists():
            row = read_json(result_path)
            if (row['protocol_sha256'] != self.protocol_hash or row['splits_sha256'] != self.splits_hash
                    or row['source_sha256'] != self.source_hashes
                    or row['checkpoint_sha256'] != ds.sha256_file(directory / 'best_model.pt')):
                raise ValueError(f'Resume pins changed: {directory}')
            return row
        directory.mkdir(parents=True, exist_ok=True)
        config = self.configuration(name, context, control)
        training = TrainConfig(seed=seed, device='cuda')
        set_reproducible_seed(seed)
        model = build_model(config)
        parameter_count = sum(p.numel() for p in model.parameters())
        start = time.perf_counter()
        print(f'START {name} context={context} seed={seed} control={control} params={parameter_count}', flush=True)
        model, history, thresholds = train_model(model, self.train, self.validation, config=training)
        metrics = evaluate_model(model, self.validation, thresholds=thresholds, batch_size=16,
                                 device='cuda', division_tolerance_frames=1)
        seconds = time.perf_counter() - start
        history_frame = pd.DataFrame(history)
        best = history_frame.loc[history_frame.validation_loss.idxmin()]
        row = {'run_id': str(directory.relative_to(self.output)).replace('\\', '/'),
               'architecture': name, 'context_frames': context, 'control': control, 'seed': seed,
               'split_seed': 42, 'split_sizes': {k: len(v) for k, v in self.splits.items()},
               'model_config': config.to_dict(), 'effective_dilations': list(config.effective_conv_dilations),
               'temporal_offsets': list(config.temporal_offsets), 'dense_support': config.has_dense_temporal_support,
               'parameter_count': parameter_count, 'train_config': training.to_dict(),
               'epochs': len(history), 'best_epoch': int(best.epoch),
               'minimum_validation_loss': float(best.validation_loss), 'seconds': seconds,
               'thresholds': {label: float(t) for label, t in zip(LABELS, thresholds)},
               'validation': metrics, 'dataset_lineage': self.lineage,
               'protocol_sha256': self.protocol_hash, 'splits_sha256': self.splits_hash,
               'source_sha256': self.source_hashes, 'completed_at_utc': utc_now(),
               'test_evaluated': False}
        row['selection_score'] = validation_score(row)
        metadata = {k: row[k] for k in ('run_id', 'context_frames', 'seed', 'split_seed', 'train_config',
                                      'dataset_lineage', 'protocol_sha256', 'splits_sha256', 'source_sha256')}
        save_checkpoint(directory / 'best_model.pt', model, model_config=config,
                        normalizer=self.normalizer, thresholds=thresholds, metadata=metadata)
        history_frame.to_csv(directory / 'history.csv', index=False)
        # Exact byte copy is the frozen split artifact, independent of training seed.
        (directory / 'splits.json').write_bytes(self.splits_path.read_bytes())
        row['checkpoint_sha256'] = ds.sha256_file(directory / 'best_model.pt')
        write_json(result_path, row)
        print(f'DONE {name} context={context} seed={seed} meanF1={row["selection_score"]["mean_f1"]:.6f} epochs={len(history)} seconds={seconds:.1f}', flush=True)
        del model
        torch.cuda.empty_cache()
        return row

    def screen(self) -> None:
        if (self.output / 'selection_lock.json').exists():
            raise ValueError('Selection is already locked; refusing new search')
        rows = [self.run(name, context, 42) for name in ('cnn', 'resnet') for context in self.dilation_map]
        control = self.run('resnet', 121, 42, True)
        assert len({r['parameter_count'] for r in rows if r['architecture'] == 'cnn'}) == 1
        assert len({r['parameter_count'] for r in rows if r['architecture'] == 'resnet'}) == 1
        write_json(self.output / 'screen_results.json', {
            'created_at_utc': utc_now(), 'protocol_sha256': self.protocol_hash, 'splits_sha256': self.splits_hash,
            'test_evaluated': False, 'rows': rows, 'control': control, 'shortlist': shortlist(rows)})

    def repeat(self) -> None:
        if (self.output / 'selection_lock.json').exists():
            raise ValueError('Selection is already locked; refusing new search')
        screen = read_json(self.output / 'screen_results.json')
        if screen['protocol_sha256'] != self.protocol_hash or screen['shortlist'] != shortlist(screen['rows']):
            raise ValueError('Screen ranking or protocol changed')
        groups = []
        for name in ('cnn', 'resnet'):
            for context in screen['shortlist'][name]:
                rows = [self.run(name, context, seed) for seed in (42, 43, 44)]
                groups.append(aggregate_rows(rows))
        winners = {name: min([r for r in groups if r['architecture'] == name], key=rank_key)
                   for name in ('cnn', 'resnet')}
        write_json(self.output / 'repeat_results.json', {'test_evaluated': False, 'groups': groups,
                                                       'protocol_sha256': self.protocol_hash})
        lock = {'locked_at_utc': utc_now(), 'protocol_sha256': self.protocol_hash,
                'screen_sha256': ds.sha256_file(self.output / 'screen_results.json'),
                'repeats_sha256': ds.sha256_file(self.output / 'repeat_results.json'),
                'split_sha256': self.splits_hash, 'test_evaluated_before_lock': False,
                'rule': self.protocol['selection'], 'winners': winners,
                'representative_seed_selected': False,
                'production_sha256': self.production_hash}
        write_json(self.output / 'selection_lock.json', lock)
        print('LOCKED', {k: v['context_frames'] for k, v in winners.items()}, flush=True)

    def boundary_metrics(self, model, dataset, thresholds, context: int) -> dict[str, Any]:
        records = collect_predictions(model, dataset, batch_size=16, device='cuda')
        by_group = {name: [] for name in ('own_context_boundary', 'own_context_interior',
                    'common_interior_radius98', 'common_boundary_radius98',
                    'trajectory_shorter_than197', 'trajectory_at_least197')}
        radius = (context - 1) // 2
        for record in records:
            n = len(record['targets']); indices = np.arange(n)
            interior = (indices >= radius) & (indices < n-radius)
            common_interior = (indices >= 98) & (indices < n-98)
            groups = (('own_context_boundary', ~interior), ('own_context_interior', interior),
                      ('common_interior_radius98', common_interior), ('common_boundary_radius98', ~common_interior),
                      ('trajectory_shorter_than197', np.full(n, n < 197)),
                      ('trajectory_at_least197', np.full(n, n >= 197)))
            for group, mask in groups:
                if mask.any():
                    by_group[group].append((record['targets'][mask], record['probabilities'][mask]))
        result = {}
        for group, pieces in by_group.items():
            if pieces:
                targets = np.concatenate([x[0] for x in pieces]); probs = np.concatenate([x[1] for x in pieces])
                result[group] = {'rows': len(targets), 'point': _point_metrics(targets, probs, np.asarray(thresholds))}
            else:
                result[group] = {'rows': 0, 'point': None}
        return result

    def evaluate(self) -> None:
        lock_path = self.output / 'selection_lock.json'
        lock = read_json(lock_path)
        if lock['protocol_sha256'] != self.protocol_hash or lock['test_evaluated_before_lock']:
            raise ValueError('Invalid selection lock')
        if lock['screen_sha256'] != ds.sha256_file(self.output / 'screen_results.json') or lock['repeats_sha256'] != ds.sha256_file(self.output / 'repeat_results.json'):
            raise ValueError('Search artifacts changed after selection')
        rows = []
        for name, winner in lock['winners'].items():
            for seed in (42,43,44):
                directory = self.run_path(name, winner['context_frames'], seed)
                row = read_json(directory / 'validation_metrics.json')
                rows.append({'label': f'{name}_rf{winner["context_frames"]}_seed{seed}', 'architecture':name,
                             'context_frames':winner['context_frames'],'seed':seed,'checkpoint':directory/'best_model.pt',
                             'checkpoint_sha256':row['checkpoint_sha256'],'selected':True})
        rows.extend([
            {'label':'R015_cnn_rf17','architecture':'cnn','context_frames':17,'seed':42,
             'checkpoint':ROOT/'runs/merged_20261029/cnn/best_model.pt','selected':False},
            {'label':'R015_resnet_rf121','architecture':'resnet','context_frames':121,'seed':42,
             'checkpoint':ROOT/'runs/merged_20261029/resnet/best_model.pt','selected':False},
            {'label':'production_R010_cnn_rf17','architecture':'cnn','context_frames':17,'seed':42,
             'checkpoint':ROOT/'src/moma_lineage_analysis/models/trajectory_cleaning_run/best_model.pt','selected':False},
        ])
        original=read_json(ROOT/'runs/merged_20261017/cnn/splits.json')
        common=sorted(set(original['test']) & set(self.splits['test']))
        conn=ds.connect_readonly(ROOT/self.protocol['dataset']['store'])
        content=dict(conn.execute('SELECT trajectory_key,content_hash FROM trajectory'));conn.close()
        seen={content[k] for split in (original,self.splits) for part in ('train','validation') for k in split[part]}
        assert not any(content[k] in seen for k in common)
        populations={'test':self.splits['test'],'common_historical_test':common,
                     'new_batch_test':[k for k in self.splits['test'] if k.startswith('curated_20261029::')]}
        datasets={name:TrajectoryDataset(self.frame,keys=keys,normalizer=self.normalizer) for name,keys in populations.items()}
        results=[]
        for row in rows:
            checkpoint=row['checkpoint']
            digest=ds.sha256_file(checkpoint)
            if row.get('checkpoint_sha256',digest)!=digest:raise ValueError('Selected checkpoint changed')
            model,normalizer,thresholds,metadata=load_checkpoint(checkpoint,device='cuda')
            results.append({**row,'checkpoint':str(checkpoint.relative_to(ROOT)).replace('\\','/'),
                            'checkpoint_sha256':digest,'thresholds':dict(zip(LABELS,map(float,thresholds))),
                            'populations':{name:evaluate_model(model,dataset,thresholds=thresholds,batch_size=16,
                                                             device='cuda',division_tolerance_frames=1)
                                           for name,dataset in datasets.items()},
                            'test_boundary':self.boundary_metrics(model,datasets['test'],thresholds,row['context_frames'])})
            print('EVALUATED',row['label'],flush=True)
            del model
        report={'selection_lock_sha256':ds.sha256_file(lock_path),'evaluation_started_after_lock':True,
                'created_at_utc':utc_now(),'selection_changed_after_test':False,
                'populations':{name:{'keys':keys,'trajectories':len(keys),
                                     'rows':int(self.frame.trajectory_key.isin(keys).sum()),
                                     'division_points':int(self.frame.loc[self.frame.trajectory_key.isin(keys),'division_point'].sum()),
                                     'excluded_points':int(self.frame.loc[self.frame.trajectory_key.isin(keys),'excluded_point'].sum())}
                               for name,keys in populations.items()},'models':results,
                'no_common_content_overlap_with_either_train_validation':True,
                'boundary_diagnostic_rule':'Use common_interior_radius98 for cross-context comparison; own_context_interior populations change with context and describe coverage only. Length strata fixed at197frames.',
                'limitations':self.protocol['reporting']['limitations'],'promotion_performed':False}
        write_json(self.output/'evaluation.json',report)
        assert ds.sha256_file(ROOT/'src/moma_lineage_analysis/models/trajectory_cleaning_run/best_model.pt')==self.production_hash


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','repeat','evaluate','all'])
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    torch.set_num_threads(2)
    sweep=Sweep(args.output_dir)
    write_json(sweep.output/'environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,
               'device':torch.cuda.get_device_name(0),'python':platform.python_version(),
               'source_sha256':sweep.source_hashes,'protocol_sha256':sweep.protocol_hash})
    stages=('screen','repeat','evaluate') if args.stage=='all' else (args.stage,)
    for stage in stages:getattr(sweep,stage)()


if __name__ == '__main__':
    main()
