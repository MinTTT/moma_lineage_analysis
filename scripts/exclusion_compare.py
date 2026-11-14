"""R022 post-lock evaluation; never use test outcomes to select a model or seed.

Usage (only after all training and validation selection are locked)::
    python scripts/exclusion_compare.py evaluate --device cuda:0

No training, promotion, store access, or checkpoint writes. Imports and pure
statistics helpers are safe before lock. The evaluate entry point verifies every
registered checkpoint before loading the private SHA-guarded frame or predicting.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import platform
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import precision_recall_curve

try:
    from . import exclusion_study as study
except ImportError:  # Direct script invocation.
    import exclusion_study as study
from moma_lineage_analysis.trajectory_cleaning.data import TrajectoryDataset
from moma_lineage_analysis.trajectory_cleaning.training import (
    _match_event_indices, best_f1_threshold, cluster_event_indices,
    collect_predictions, load_checkpoint,
)

ROOT, OUT, PRODUCTION = study.ROOT, study.OUT, study.PRODUCTION
METRICS = ('exclusion_ap', 'exclusion_precision', 'exclusion_recall',
           'exclusion_f1', 'division_event_f1')
BOOTSTRAP_SEED = 20261110
PANELS = ('all_clean_test', 'historical_clean_test', 'new_batch_clean_test')
LIMITATIONS = [
    'Historical validation and test observations have been reused; descriptive, not fresh confirmatory evidence.',
    'Trajectory-grouped split and trajectory bootstrap do not establish FOV independence or new-FOV generalization.',
    'Seed SD measures optimization noise; the three seeds are not independent biological replicates (n=3).',
    'Intervals are paired trajectory-cluster percentile 95% intervals, without multiplicity correction.',
    'Seed42 was fixed before evaluation; no test-based model, threshold, or seed selection.',
    'New-batch support is very small; missing AP and no-positive bootstrap replicates are explicitly reported.',
    'Production deployed thresholds are the primary control; R022-validation recalibration is a separate sensitivity control.',
    'No deployment or promotion is performed or implied.',
]


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _pin(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(f'Frozen artifact hash mismatch: {path}')


def _thresholds(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Exactly two finite label thresholds in [0,1] required')
    return values


def _val_values(record):
    values = dict(exclusion_ap=record['validation']['point']['excluded_point']['average_precision'],
                  exclusion_f1=record['validation']['point']['excluded_point']['f1'],
                  division_event_f1=record['validation']['division_event']['f1'])
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1
           for v in values.values()):
        raise ValueError('Invalid validation selection metric')
    return values


def validate_lock(output=OUT, production=None):
    """Read-only gate. All 21 hashes freeze losing arms too; no model is loaded.

    Returns protocol, lock, validation records, and an input-hash manifest. The
    study's lock schema records per-seed checkpoint hashes, not filesystem flags;
    immutability here means complete registration and matching content hashes.
    """
    output, production = Path(output), Path(PRODUCTION if production is None else production)
    lock_path = output / 'selection_lock.json'
    if not lock_path.is_file():
        raise ValueError('Test inference refused: selection_lock.json is required')
    lock = study.read_json(lock_path)
    protocol_path = output / 'protocol.json'
    _pin(protocol_path, lock.get('protocol_sha256'))
    protocol = study.read_json(protocol_path)
    seeds, arms = protocol['seeds'], list(protocol['arms'])
    if seeds != [42, 43, 44] or set(arms) != set(study.ARMS):
        raise ValueError('Expected registered seven arms and seeds42/43/44')
    if (lock.get('test_evaluated_before_lock') is not False
            or lock.get('selection_uses') != 'validation only'
            or lock.get('representative_seed') != 42 or lock.get('seeds') != seeds):
        raise ValueError('Invalid validation-only selection lock or representative seed')
    rows = lock.get('rows', [])
    if len(rows) != len(arms) or {r['arm'] for r in rows} != set(arms):
        raise ValueError('Lock must freeze every registered arm exactly once')
    pins = {str(lock_path): sha256(lock_path), str(protocol_path): sha256(protocol_path),
            str(Path(__file__).resolve()): sha256(__file__)}
    for path, expected in [(output / 'frame.pkl', protocol['dataset_snapshot_sha256']),
                           (output / 'splits.json', protocol['split_sha256']),
                           (production / 'best_model.pt', study.PRODUCTION_SHA),
                           (production / 'splits.json', protocol['production_splits_sha256'])]:
        _pin(path, expected)
        pins[str(path)] = expected
    if protocol['production_sha256'] != study.PRODUCTION_SHA:
        raise ValueError('Protocol production pin differs from released R019 pin')
    validations, means = {}, {}
    for row in rows:
        arm = row['arm']
        if row.get('seeds') != seeds or set(row.get('checkpoint_sha256', {})) != {str(s) for s in seeds}:
            raise ValueError(f'Incomplete frozen checkpoint inventory: {arm}')
        values = []
        for seed in seeds:
            run = output / 'training' / arm / f'seed{seed}'
            checkpoint = run / 'best_model.pt'
            expected = row['checkpoint_sha256'][str(seed)]
            _pin(checkpoint, expected)
            path = run / 'validation.json'
            record = study.read_json(path)
            if (record.get('arm') != arm or record.get('seed') != seed
                    or record.get('test_evaluated') is not False
                    or record.get('protocol_sha256') != lock['protocol_sha256']
                    or record.get('checkpoint_sha256') != expected):
                raise ValueError(f'Validation provenance mismatch: {arm}/seed{seed}')
            _thresholds(record['thresholds'])
            _pin(run / 'splits.json', protocol['split_sha256'])
            source_hashes = record.get('source_hashes', {})
            if not set(study.SOURCE_PATHS).issubset(source_hashes):
                raise ValueError('Training source hash inventory is incomplete')
            for relative, source_hash in source_hashes.items():
                source_path = ROOT / relative
                _pin(source_path, source_hash)
                pins[str(source_path)] = source_hash
            values.append(_val_values(record))
            validations[(arm, seed)] = record
            pins[str(checkpoint)] = expected
            pins[str(path)] = sha256(path)
            pins[str(run / 'splits.json')] = protocol['split_sha256']
        mean = {m: statistics.mean(v[m] for v in values) for m in values[0]}
        sd = {m: statistics.stdev(v[m] for v in values) for m in values[0]}
        if any(not math.isclose(row['mean'][m], mean[m], abs_tol=1e-12, rel_tol=0)
               or not math.isclose(row['seed_sd'][m], sd[m], abs_tol=1e-12, rel_tol=0)
               for m in mean):
            raise ValueError(f'Locked validation aggregate mismatch: {arm}')
        means[arm] = mean
    base = means['baseline']
    for row in rows:
        mean = means[row['arm']]
        eligible = (mean['division_event_f1'] >= base['division_event_f1'] - .002
                    and mean['exclusion_f1'] >= base['exclusion_f1'] - .01)
        if row.get('eligible') is not eligible:
            raise ValueError('Locked eligibility disagrees with guardrails')
    ranked = sorted((r['arm'] for r in rows if r['eligible']), key=lambda a: (
        -means[a]['exclusion_ap'], -means[a]['exclusion_f1'],
        -means[a]['division_event_f1'], arms.index(a)))
    if lock.get('ranked_eligible') != ranked or lock.get('selected_arm') != ranked[0]:
        raise ValueError('Locked winner disagrees with validation-only ranking')
    return protocol, lock, validations, pins


def verify_pins(pins):
    for path, expected in pins.items():
        _pin(path, expected)


def canonical_rows(frame):
    """Whole trajectories, key/time stable; preserve input order for equal times."""
    frame = frame.copy()
    frame['trajectory_key'] = frame.trajectory_key.astype(str)
    frame['fov_cell_index'] = frame.fov_cell_index.astype(str)
    return frame.sort_values(['trajectory_key', 'time_h'], kind='stable').reset_index(drop=True)


def clean_panels(frame, test_keys, production_train_keys):
    """Compare exact time/length signals against every original R019 TRAIN key.

    The frozen snapshot retains all original production train keys. Refuse a
    partial inventory rather than infer that an omitted train signal was clean.
    IDs (including FOV IDs) and labels are deliberately absent from signatures.
    """
    present = set(frame.trajectory_key.astype(str))
    train = set(map(str, production_train_keys))
    tests = set(map(str, test_keys))
    if not train.issubset(present) or not tests.issubset(present):
        raise ValueError('Frozen frame lacks original production train or R022 test keys')
    signature_keys = study.signatures(frame)
    lookup = {key: signature for signature, keys in signature_keys.items() for key in keys}
    train_signals = {}
    for key in sorted(train):
        train_signals.setdefault(lookup[key], []).append(key)
    dropped, retained = [], []
    for key in sorted(tests):
        matches = train_signals.get(lookup[key], [])
        if matches:
            dropped.append(dict(trajectory_key=key, signal_sha256=lookup[key],
                production_train_keys=';'.join(matches), reason='exact time/length twin of R019 TRAIN (IDs ignored)'))
        else:
            retained.append(key)
    sources = frame.groupby('trajectory_key').source_dataset.agg(lambda x: set(x.astype(str)))
    if any(len(sources[k]) != 1 for k in retained):
        raise ValueError('Trajectory source_dataset is not unique')
    new = [k for k in retained if sources[k] == {'curated_20261109'}]
    historical = [k for k in retained if k not in set(new)]
    return dict(zip(PANELS, [retained, historical, new])), dropped


class ClusterAP:
    """Exact weighted AP, sorting once and reducing exact tied-score groups.

    Cluster multiplicities weight every row of a trajectory identically. This
    equals sklearn average_precision_score(..., sample_weight=row_weights), but
    avoids sorting and sklearn setup for every bootstrap replicate.
    """
    def __init__(self, target, probability, cluster):
        target, probability, cluster = np.asarray(target), np.asarray(probability), np.asarray(cluster)
        if not (target.ndim == probability.ndim == cluster.ndim == 1
                and target.shape == probability.shape == cluster.shape):
            raise ValueError('AP inputs must be aligned one-dimensional arrays')
        if not np.isfinite(probability).all() or not np.isin(target, [0, 1]).all():
            raise ValueError('AP requires finite probabilities and binary targets')
        order = np.argsort(-probability, kind='stable')
        self.target = target[order].astype(float)
        self.cluster = cluster[order].astype(np.int64)
        sorted_probability = probability[order]
        self.starts = np.r_[0, 1 + np.flatnonzero(np.diff(sorted_probability))] if len(order) else np.array([], dtype=int)

    def __call__(self, multiplicities):
        if not len(self.starts):
            return float('nan')
        weight = np.asarray(multiplicities, dtype=float)[self.cluster]
        group_positive = np.add.reduceat(weight * self.target, self.starts)
        total_positive = group_positive.sum()
        if total_positive == 0:
            return float('nan')
        group_total = np.add.reduceat(weight, self.starts)
        cumulative_total = np.cumsum(group_total)
        precision = np.divide(np.cumsum(group_positive), cumulative_total,
                              out=np.zeros_like(cumulative_total), where=cumulative_total > 0)
        return float(np.sum(group_positive * precision) / total_positive)


def cluster_multiplicities(rng, count):
    """Draw count complete trajectories with replacement; never sample rows."""
    return np.bincount(rng.integers(0, count, size=count), minlength=count) if count else np.zeros(0, dtype=int)


def _rates(tp, fp, fn):
    return (float(tp / max(tp + fp, 1)), float(tp / max(tp + fn, 1)),
            float(2 * tp / max(2 * tp + fp + fn, 1)))


class PanelStatistics:
    def __init__(self, table, thresholds):
        thresholds = _thresholds(thresholds)
        self.table = canonical_rows(table)
        self.keys = self.table.trajectory_key.unique().tolist()
        cluster = pd.Categorical(self.table.trajectory_key, categories=self.keys).codes
        target = self.table.excluded_point.to_numpy(dtype=bool)
        probability = self.table.exclusion_probability.to_numpy(dtype=float)
        self.ap = ClusterAP(target, probability, cluster)
        counts = []
        for _, group in self.table.groupby('trajectory_key', sort=False):
            y = group.excluded_point.to_numpy(dtype=bool)
            pred = group.exclusion_probability.to_numpy() >= thresholds[1]
            truth = cluster_event_indices(group.division_point.to_numpy() >= .5, excluded=y)
            div_p = group.division_probability.to_numpy()
            predicted = cluster_event_indices(div_p >= thresholds[0], score=div_p)
            tp, fp, fn, _ = _match_event_indices(truth, predicted, 1)
            counts.append([np.sum(y & pred), np.sum(~y & pred), np.sum(y & ~pred),
                           np.sum(~y & ~pred), tp, fp, fn, len(group), np.sum(y)])
        self.counts = np.asarray(counts, dtype=np.int64).reshape((-1, 9))

    def metrics(self, multiplicities=None):
        multiplicities = np.ones(len(self.keys), dtype=int) if multiplicities is None else np.asarray(multiplicities)
        if multiplicities.shape != (len(self.keys),) or (multiplicities < 0).any():
            raise ValueError('One nonnegative multiplicity per complete trajectory required')
        tp, fp, fn, tn, dtp, dfp, dfn, rows, positive = multiplicities @ self.counts
        precision, recall, f1 = _rates(tp, fp, fn)
        dp, dr, df1 = _rates(dtp, dfp, dfn)
        nonempty = rows > 0
        return dict(exclusion_ap=self.ap(multiplicities),
            exclusion_precision=precision if nonempty else float('nan'),
            exclusion_recall=recall if nonempty else float('nan'),
            exclusion_f1=f1 if nonempty else float('nan'),
            division_event_f1=df1 if nonempty else float('nan'),
            division_event_precision=dp if nonempty else float('nan'),
            division_event_recall=dr if nonempty else float('nan'),
            exclusion_tp=int(tp), exclusion_fp=int(fp), exclusion_fn=int(fn), exclusion_tn=int(tn),
            division_event_tp=int(dtp), division_event_fp=int(dfp), division_event_fn=int(dfn),
            rows=int(rows), exclusions=int(positive), trajectories=int(multiplicities.sum()),
            exclusion_ap_status='defined' if positive else 'undefined_no_positive',
            panel_status='nonempty' if nonempty else 'empty')


def assert_paired(left, right):
    columns = ['trajectory_key', 'fov_cell_index', 'time_h', 'spine_length',
               'division_point', 'excluded_point', 'source_dataset']
    if not left[columns].reset_index(drop=True).equals(right[columns].reset_index(drop=True)):
        raise ValueError('Models must share identical keyed rows, order, signals, sources and targets')


def paired_bootstrap(left, right, left_thresholds, right_thresholds, *, replicates=5000, seed=BOOTSTRAP_SEED):
    """Candidate minus reference on paired whole-trajectory draws within a panel.

    Precision/recall/F1 use the package zero_division=0 convention; AP without
    positives is undefined. Intervals use only finite replicates and disclose all
    undefined and no-positive counts. Empty panels have no defined estimates.
    """
    if replicates < 1:
        raise ValueError('Bootstrap replicate count must be positive')
    left, right = canonical_rows(left), canonical_rows(right)
    assert_paired(left, right)
    a, b = PanelStatistics(left, left_thresholds), PanelStatistics(right, right_thresholds)
    point_a, point_b = a.metrics(), b.metrics()
    draws = np.full((replicates, len(METRICS)), np.nan)
    rng = np.random.default_rng(seed)
    no_positive = 0
    for i in range(replicates):
        multiplicities = cluster_multiplicities(rng, len(a.keys))
        am, bm = a.metrics(multiplicities), b.metrics(multiplicities)
        no_positive += am['exclusions'] == 0
        draws[i] = [am[m] - bm[m] for m in METRICS]
    result = []
    for j, metric in enumerate(METRICS):
        valid = draws[np.isfinite(draws[:, j]), j]
        low, high = np.percentile(valid, [2.5, 97.5]) if len(valid) else (float('nan'), float('nan'))
        result.append(dict(metric=metric, point_delta=point_a[metric] - point_b[metric],
            ci_lower=float(low), ci_upper=float(high), replicates_requested=replicates,
            replicates_valid=len(valid), replicates_undefined=replicates-len(valid),
            no_positive_replicates=int(no_positive), bootstrap_seed=seed,
            trajectories=len(a.keys), rows=len(left), exclusions=point_a['exclusions'],
            interval='paired trajectory-cluster percentile 95%', multiplicity_correction='none'))
    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _predict(checkpoint, frame, keys, device, *, expected_hash, expected_record=None):
    """Private inference adapter; called only after evaluate's complete lock gate."""
    _pin(checkpoint, expected_hash)
    model, normalizer, thresholds, metadata = load_checkpoint(checkpoint, device=device)
    thresholds = _thresholds(thresholds)
    if expected_record is not None:
        if (metadata.get('arm') != expected_record['arm'] or metadata.get('seed') != expected_record['seed']
                or metadata.get('protocol_sha256') != expected_record['protocol_sha256']
                or metadata.get('test_evaluated') is not False
                or metadata.get('source_hashes') != expected_record['source_hashes']):
            raise ValueError('Checkpoint metadata differs from frozen validation record')
        if not np.array_equal(thresholds.astype(np.float32), _thresholds(expected_record['thresholds']).astype(np.float32)):
            raise ValueError('Checkpoint thresholds differ from validation-calibrated thresholds')
    table = canonical_rows(frame[frame.trajectory_key.isin(keys)])
    if table.empty:
        table['division_probability'] = pd.Series(dtype=float)
        table['exclusion_probability'] = pd.Series(dtype=float)
        return table, thresholds
    dataset = TrajectoryDataset(table, normalizer=normalizer)
    records = collect_predictions(model, dataset, device=device, batch_size=16)
    probability = np.full((len(table), 2), np.nan)
    seen = np.zeros(len(table), dtype=np.int64)
    targets = table[['division_point', 'excluded_point']].to_numpy()
    for record in records:
        positions = np.asarray(record['row_positions'], dtype=int)
        if not np.array_equal(targets[positions], record['targets']):
            raise ValueError('Inference target alignment failed')
        if not (table.iloc[positions].trajectory_key == str(record['key'])).all():
            raise ValueError('Inference key alignment failed')
        probability[positions] = record['probabilities']
        seen[positions] += 1
    if not (seen == 1).all() or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Every row must have exactly one finite probability pair in [0,1]')
    table['division_probability'], table['exclusion_probability'] = probability.T
    _pin(checkpoint, expected_hash)
    return table, thresholds


def _write_csv(path, table):
    pd.DataFrame(table).to_csv(path, index=False, float_format='%.17g')


def validation_table(lock, validations):
    return pd.DataFrame([dict(arm=row['arm'], seed=seed, eligible=row['eligible'],
        selected=row['arm'] == lock['selected_arm'], **_val_values(validations[(row['arm'], seed)]),
        mean_exclusion_ap=row['mean']['exclusion_ap'], seed_sd_exclusion_ap=row['seed_sd']['exclusion_ap'])
        for row in lock['rows'] for seed in lock['seeds']])


def summarize_seeds(metrics):
    results = []
    for (panel, arm), group in metrics[metrics.seed.notna()].groupby(['panel', 'arm'], sort=False):
        row = dict(panel=panel, arm=arm, seeds='42;43;44', seed_count=len(group),
                   seed_sd_interpretation='optimization noise; not independent biological evidence')
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            valid = values[np.isfinite(values)]
            row[metric + '_mean'] = float(np.mean(valid)) if len(valid) else float('nan')
            row[metric + '_seed_sd'] = float(np.std(valid, ddof=1)) if len(valid) > 1 else float('nan')
            row[metric + '_valid_seeds'] = len(valid)
        results.append(row)
    return pd.DataFrame(results)


def comparison_table(metrics):
    """Wide side-by-side controls, all seeds, and mean/seed SD for each metric."""
    result = []
    for panel, group in metrics.groupby('panel', sort=False):
        for metric in METRICS:
            row = dict(panel=panel, metric=metric, trajectories=int(group.iloc[0].trajectories),
                       rows=int(group.iloc[0].rows), exclusions=int(group.iloc[0].exclusions))
            for _, record in group.iterrows():
                row[record.model_id] = record[metric]
            for arm, seeded in group[group.seed.notna()].groupby('arm', sort=False):
                values = seeded[metric].to_numpy(dtype=float)
                valid = values[np.isfinite(values)]
                row[arm+'_mean'] = float(np.mean(valid)) if len(valid) else float('nan')
                row[arm+'_seed_sd'] = float(np.std(valid, ddof=1)) if len(valid)>1 else float('nan')
                row[arm+'_valid_seeds'] = len(valid)
            result.append(row)
    return pd.DataFrame(result)


def render_figures(destination, validation, metrics, tables, panels, *, bootstrap=None, synthetic=False):
    """Agg-only provisional report figures, all source tables and missing states explicit."""
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from PIL import Image

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / 'validation_selection_source.csv', validation)
    _write_csv(destination / 'test_metrics_source.csv', metrics)
    curves = []
    style = {'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
             'pdf.fonttype': 42, 'figure.facecolor': 'white', 'axes.facecolor': 'white'}
    figure_metadata = []
    watermark = 'SYNTHETIC CHECK ONLY — ' if synthetic else ''
    def save(fig, name, description):
        for extension in ['png', 'pdf']:
            path = destination / f'{name}.{extension}'
            fig.savefig(path, dpi=200, facecolor='white', transparent=False)
            info = dict(file=path.name, sha256=sha256(path), bytes=path.stat().st_size)
            if extension == 'png':
                with Image.open(path) as image:
                    info.update(pixels=list(image.size), dpi=list(image.info.get('dpi', [])), mode=image.mode)
            figure_metadata.append(info)
        plt.close(fig)
        return dict(name=name, description=description)
    descriptions = []
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
        arms = validation.arm.drop_duplicates().tolist()
        for x, arm in enumerate(arms):
            group = validation[validation.arm == arm]
            eligible, selected = bool(group.eligible.iloc[0]), bool(group.selected.iloc[0])
            color = '#0072B2' if eligible else '#777777'
            ax.scatter(x + np.linspace(-.18, .06, len(group)), group.exclusion_ap,
                       marker='o' if eligible else 'x', c=color, s=35)
            ax.scatter([x + .22], [group.mean_exclusion_ap.iloc[0]], marker='D',
                       facecolors='white', edgecolors='black', s=65, zorder=4)
            ax.text(x, .025, ('SELECTED\n' if selected else '') + ('eligible' if eligible else 'ineligible'),
                    ha='center', fontsize=8)
        ax.set(xticks=range(len(arms)), xticklabels=arms, ylim=(0, 1), ylabel='Validation exclusion AP',
               title=watermark + 'Validation-only selection: all seed points; diamonds = mean')
        ax.tick_params(axis='x', labelrotation=18)
        fig.supxlabel('Guardrails: mean division-event F1 ≥ baseline − 0.002; mean exclusion F1 ≥ baseline − 0.01')
        descriptions.append(save(fig, 'validation_selection', 'All registered arms, all three seed APs and means; eligibility uses both guardrails. Full 0–1 scale.'))
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
        for ax, panel in zip(axes, PANELS):
            keys = panels[panel]
            displayed_tables = [(name, table) for name, table in tables.items()
                                if name == 'production_deployed' or name.endswith('seed42')]
            for i, (model_id, table) in enumerate(displayed_tables):
                sub = table[table.trajectory_key.isin(keys)]
                if not len(sub) or not sub.excluded_point.any():
                    ax.text(.02, .12 + .08*i, f'{model_id}: AP undefined (no positives)', transform=ax.transAxes, fontsize=8)
                    continue
                precision, recall, thresholds = precision_recall_curve(sub.excluded_point, sub.exclusion_probability)
                ax.step(recall, precision, where='post', label=model_id,
                        color=['#222222', '#0072B2', '#D55E00'][i % 3], linestyle=['-', '--', ':'][i % 3])
                for p, r, t in zip(precision, recall, np.r_[thresholds, np.nan]):
                    curves.append(dict(panel=panel, model_id=model_id, precision=p, recall=r, threshold=t))
            support = metrics[metrics.panel == panel].iloc[0]
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Exclusion recall', ylabel='Exclusion precision',
                   title=f'{panel}\n{int(support.trajectories)} trajectories / {int(support.rows)} rows / {int(support.exclusions)} exclusions')
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8, loc='lower left')
            if panel == 'new_batch_clean_test':
                ax.text(.5, .95, 'VERY SMALL SUPPORT: descriptive only', transform=ax.transAxes, ha='center', fontsize=8)
        fig.suptitle(watermark + 'Fixed seed42 and deployed production: identical rows; no threshold selection on test')
        descriptions.append(save(fig, 'test_precision_recall', 'Unsmooth exclusion precision–recall curves for fixed seed42 and production on three paired panels; small and undefined support labeled.'))
        _write_csv(destination / 'precision_recall_source.csv', pd.DataFrame(curves, columns=['panel', 'model_id', 'precision', 'recall', 'threshold']))
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), layout='constrained')
        displayed = [('exclusion_f1', 'Exclusion F1'), ('exclusion_ap', 'Exclusion AP'), ('division_event_f1', 'Division-event F1 (±1 frame)')]
        for row_index, panel in enumerate(PANELS):
            sub = metrics[metrics.panel == panel]
            groups = sub.arm.drop_duplicates().tolist()
            support = sub.iloc[0]
            for ax, (metric, label) in zip(axes[row_index], displayed):
                for x, arm in enumerate(groups):
                    group = sub[sub.arm == arm]
                    for j, (_, entry) in enumerate(group.iterrows()):
                        value = entry[metric]
                        if pd.isna(value):
                            ax.text(x, .5, 'undefined', rotation=90, ha='center', color='#666666')
                        else:
                            ax.scatter(x + (j - (len(group)-1)/2)*.13, value,
                                marker='s' if pd.isna(entry.seed) else {42:'o',43:'^',44:'v'}[int(entry.seed)],
                                color=['#222222', '#777777', '#0072B2', '#D55E00'][x % 4], s=42)
                ax.set(xticks=range(len(groups)), xticklabels=groups, ylim=(-.025, 1.025),
                       yticks=np.linspace(0, 1, 6), ylabel=label)
                ax.tick_params(axis='x', labelrotation=20, labelsize=8)
                ax.set_title(f'{panel}: {int(support.trajectories)} trajectories; {int(support.exclusions)} exclusions', fontsize=10)
                if panel == 'new_batch_clean_test':
                    ax.text(.5, .08, 'VERY SMALL SUPPORT', transform=ax.transAxes, ha='center', fontsize=9)
        fig.suptitle(watermark + 'Test point metrics: circles seed42; triangles seeds43/44; squares production controls')
        fig.supxlabel('Seed spread measures optimization noise, not biological replication. Primary seed42 sampling intervals are reported separately.')
        descriptions.append(save(fig, 'test_seed_points', 'All seed point metrics plus both production threshold controls; full 0–1 scale with small display margins so boundary markers remain visible.'))
        if bootstrap is not None:
            intervals = pd.DataFrame(bootstrap)
            fig, axes = plt.subplots(3, 5, figsize=(23, 10), layout='constrained')
            for row_index, panel in enumerate(PANELS):
                support = metrics[metrics.panel == panel].iloc[0]
                for ax, metric in zip(axes[row_index], METRICS):
                    sub = intervals[(intervals.panel == panel) & (intervals.metric == metric)]
                    for y, (_, entry) in enumerate(sub.iterrows()):
                        label = ('baseline' if entry.model_id.startswith('baseline_') else 'selected')
                        label += ' − ' + ('production' if entry.reference == 'production_deployed' else 'baseline')
                        point, low, high = entry.point_delta, entry.ci_lower, entry.ci_upper
                        if np.isfinite(point) and np.isfinite(low) and np.isfinite(high):
                            ax.plot([low, high], [y, y], color='#0072B2', linewidth=2)
                            ax.scatter(point, y, marker='o', color='#222222', s=28, zorder=3)
                        else:
                            ax.text(0, y, 'undefined', ha='center', va='center', fontsize=8)
                        ax.text(-.98, y+.18, f"{int(entry.replicates_valid)}/{int(entry.replicates_requested)} valid; {int(entry.no_positive_replicates)} no-positive", fontsize=7)
                    ax.axvline(0, color='#777777', linestyle='--', linewidth=.8)
                    labels = [('baseline' if e.model_id.startswith('baseline_') else 'selected') + ' − ' +
                              ('production' if e.reference=='production_deployed' else 'baseline') for _,e in sub.iterrows()]
                    ax.set(xlim=(-1,1), ylim=(-.6, max(len(sub)-.3,.6)), yticks=range(len(sub)),
                           yticklabels=labels, xlabel='Candidate − reference', title=metric.replace('_', ' '))
                    ax.tick_params(labelsize=8)
                    if row_index == 0:
                        ax.set_title(metric.replace('_', ' '), fontsize=10)
                axes[row_index, 0].set_ylabel(f'{panel}\n{int(support.trajectories)} trajectories / {int(support.exclusions)} exclusions', fontsize=9)
            fig.suptitle(watermark + 'Fixed seed42: paired trajectory-cluster percentile 95% intervals; no multiplicity correction')
            fig.supxlabel('Same whole-trajectory draws for each pair; AP excludes no-positive replicates. Historical test reuse; new-batch support is very small.')
            descriptions.append(save(fig, 'seed42_paired_intervals', 'Candidate minus reference point deltas and paired trajectory-cluster percentile 95% intervals; full −1..1 difference scale, valid and no-positive replicate counts disclosed.'))
    if bootstrap is not None:
        _write_csv(destination / 'bootstrap_interval_source.csv', bootstrap)
    sources = {p.name: sha256(p) for p in destination.glob('*_source.csv')}
    return dict(audience='provisional scientific report; no target journal specified',
        synthetic=synthetic, backend='Agg', matplotlib=matplotlib.__version__, numpy=np.__version__,
        figure_files=figure_metadata, figures=descriptions, source_sha256=sources,
        transformations=['frozen test-key membership', 'drop exact time/length twins of R019 TRAIN',
            'per-label validation thresholds', 'full 0..1 metric axes', 'no smoothing'],
        uncertainty='Seed points are optimization noise; primary seed42 paired trajectory-cluster CI is in bootstrap_interval_source.csv',
        limitations=LIMITATIONS, publisher_requirements='pending; general report figures only')


def evaluate(output=OUT, *, device='cuda:0', replicates=5000):
    """The sole evaluation command. Lock gate precedes frame load and inference."""
    output = Path(output)
    protocol, lock, validations, pins = validate_lock(output)
    if device not in ('cuda:0', 'cpu'):
        raise ValueError('Evaluation permits GPU0 (cuda:0) or explicit cpu only; GPU1 replay is prohibited')
    if replicates != 5000:
        raise ValueError('Registered actual evaluation requires exactly 5000 bootstrap replicates')
    destination = output / 'evaluation'
    if destination.exists():
        raise ValueError('Evaluation directory exists; preserve it, do not overwrite predictions or reports')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    loaded_protocol, frame, splits = study.load_study(output)
    if loaded_protocol != protocol:
        raise ValueError('Protocol changed after gate')
    original = study.read_json(PRODUCTION / 'splits.json')
    panels, dropped = clean_panels(frame, splits['test'], original['train'])
    selected_arms = list(dict.fromkeys(['baseline', lock['selected_arm']]))
    tables, thresholds, identities, checkpoint_pins = {}, {}, {}, {}
    clean_keys = panels['all_clean_test']
    # Production is predicted once on validation + complete clean test trajectories;
    # both threshold controls reuse precisely the same probabilities.
    combined_keys = list(dict.fromkeys(splits['validation'] + clean_keys))
    verify_pins(pins)
    production_table, saved_thresholds = _predict(PRODUCTION / 'best_model.pt', frame, combined_keys,
        device, expected_hash=protocol['production_sha256'])
    validation_predictions = production_table[production_table.trajectory_key.isin(splits['validation'])]
    recalibrated = np.array([best_f1_threshold(validation_predictions[label].to_numpy(),
        validation_predictions[probability].to_numpy()) for label, probability in
        [('division_point', 'division_probability'), ('excluded_point', 'exclusion_probability')]])
    production_test = canonical_rows(production_table[production_table.trajectory_key.isin(clean_keys)])
    for name, values in [('production_deployed', saved_thresholds), ('production_recalibrated', recalibrated)]:
        tables[name], thresholds[name] = production_test.copy(), values
        identities[name] = dict(arm=name, seed=None, threshold_origin='deployed R019 checkpoint' if name.endswith('deployed') else 'R022 validation only; independent point-F1 grid per label')
        checkpoint_pins[name] = protocol['production_sha256']
    for arm in selected_arms:
        for seed in protocol['seeds']:
            name = f'{arm}_seed{seed}'
            record = validations[(arm, seed)]
            verify_pins(pins)
            # Full trajectories only; losing arms never enter this inference path.
            table, values = _predict(output / 'training' / arm / f'seed{seed}' / 'best_model.pt',
                frame, clean_keys, device, expected_hash=record['checkpoint_sha256'], expected_record=record)
            assert_paired(production_test, table)
            tables[name], thresholds[name] = table, values
            identities[name] = dict(arm=arm, seed=seed, threshold_origin='own R022 validation-calibrated checkpoint')
            checkpoint_pins[name] = record['checkpoint_sha256']
    verify_pins(pins)
    destination.mkdir()
    provenance = dict(protocol_sha256=lock['protocol_sha256'], selection_lock_sha256=pins[str(output/'selection_lock.json')],
        frame_sha256=protocol['dataset_snapshot_sha256'], splits_sha256=protocol['split_sha256'])
    for name, table in tables.items():
        table['model_id'] = name
        table['checkpoint_sha256'] = checkpoint_pins[name]
        for field, value in provenance.items():
            table[field] = value
        table['division_threshold'], table['exclusion_threshold'] = thresholds[name]
        _write_csv(destination / f'rows_{name}.csv', table)
    # Archive the production calibration data separately; never derive thresholds from test.
    calibration_archive = validation_predictions.assign(model_id='production_validation_calibration',
        checkpoint_sha256=protocol['production_sha256'], **provenance)
    _write_csv(destination / 'production_validation_calibration_rows.csv', calibration_archive)
    metric_rows, delta_rows, boot_rows = [], [], []
    for panel, keys in panels.items():
        subsets = {name: table[table.trajectory_key.isin(keys)] for name, table in tables.items()}
        points = {name: PanelStatistics(table, thresholds[name]).metrics() for name, table in subsets.items()}
        for name, point in points.items():
            metric_rows.append(dict(panel=panel, model_id=name, **identities[name],
                division_threshold=thresholds[name][0], exclusion_threshold=thresholds[name][1], **point))
        for arm in selected_arms:
            for seed in protocol['seeds']:
                name = f'{arm}_seed{seed}'
                references = ['production_deployed'] + ([f'baseline_seed{seed}'] if arm != 'baseline' else [])
                for reference in references:
                    for metric in METRICS:
                        delta_rows.append(dict(panel=panel, model_id=name, reference=reference, seed=seed,
                            metric=metric, point_delta=points[name][metric]-points[reference][metric]))
                    if seed == 42:
                        print(f'Bootstrap {panel}: {name} minus {reference} ({replicates} replicates)', flush=True)
                        boot_rows.extend(dict(panel=panel, model_id=name, reference=reference, seed=seed, **r)
                            for r in paired_bootstrap(subsets[name], subsets[reference], thresholds[name], thresholds[reference], replicates=replicates))
    metrics, validation = pd.DataFrame(metric_rows), validation_table(lock, validations)
    _write_csv(destination / 'point_metrics.csv', metrics)
    _write_csv(destination / 'seed_summary.csv', summarize_seeds(metrics))
    _write_csv(destination / 'comparison_table.csv', comparison_table(metrics))
    _write_csv(destination / 'point_deltas.csv', delta_rows)
    _write_csv(destination / 'bootstrap_seed42.csv', boot_rows)
    _write_csv(destination / 'dropped_test_keys.csv', pd.DataFrame(dropped,
        columns=['trajectory_key', 'signal_sha256', 'production_train_keys', 'reason']))
    membership = [dict(panel=p, trajectory_key=k) for p, keys in panels.items() for k in keys]
    _write_csv(destination / 'panel_membership.csv', pd.DataFrame(membership, columns=['panel','trajectory_key']))
    figures = render_figures(destination, validation, metrics, tables, panels, bootstrap=boot_rows)
    verify_pins(pins)
    files = {p.name: sha256(p) for p in destination.iterdir() if p.is_file()}
    report = dict(round='R022', evaluated_at=study.now(), status='descriptive post-lock comparison; no promotion',
        selected_arm=lock['selected_arm'], evaluated_arms=selected_arms, seeds=protocol['seeds'], primary_seed=42,
        primary_control='production_deployed', device=device,
        device_caveat='CPU replay may drift from GPU0; GPU1 was not used' if device == 'cpu' else 'GPU0 inference',
        threads=2, python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        pandas=pd.__version__, sklearn=sklearn.__version__, platform=platform.platform(),
        bootstrap=dict(replicates=replicates, seed=BOOTSTRAP_SEED, unit='whole trajectory', confidence=.95,
            method='paired percentile', no_positive_ap='undefined, excluded from percentiles with counts recorded',
            zero_division='precision/recall/F1 = 0; empty panel = undefined', multiplicity_correction='none'),
        production_thresholds=saved_thresholds.tolist(), production_recalibrated_thresholds=recalibrated.tolist(),
        population=protocol['population'], panels={p:dict(keys=keys, **PanelStatistics(
            production_test[production_test.trajectory_key.isin(keys)], saved_thresholds).metrics()) for p,keys in panels.items()},
        dropped_keys=dropped, preparation_duplicate_exclusions=protocol.get('duplicate_exclusions', []),
        input_sha256=pins, artifacts_sha256=files, limitations=LIMITATIONS)
    study.write_json(destination / 'comparison.json', _json_safe(report))
    figures.update(provenance, input_sha256=pins, generated_at=study.now())
    study.write_json(destination / 'figure_provenance.json', _json_safe(figures))
    print(f'Comparison saved to {destination}; no promotion performed.', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate'])
    parser.add_argument('--output', type=Path, default=OUT, help='Frozen study directory (not report destination)')
    parser.add_argument('--device', default='cuda:0', choices=['cuda:0', 'cpu'])
    args = parser.parse_args()
    evaluate(args.output, device=args.device)


if __name__ == '__main__':
    main()
