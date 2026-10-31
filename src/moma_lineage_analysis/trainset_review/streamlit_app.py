"""Focused review, annotation and evaluation of training trajectories."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from moma_lineage_analysis.model_registry import AI_MODEL_REGISTRY
import streamlit.components.v1 as components
from moma_lineage_analysis.trajectory_view_style import shortcut_script
from moma_lineage_analysis.trainset_review import evaluation, plots, review, store

st.set_page_config(page_title='Training Dataset Review', page_icon='🔎', layout='wide')
st.markdown("""<style>
[data-testid="stToastContainer"] {
  left: 1rem !important; right: auto !important;
  bottom: 1rem !important; top: auto !important;
  width: auto !important; align-items: flex-start !important;
}
[data-testid="stToast"] {
  width: min(320px, calc(100vw - 2rem)) !important;
}
</style>""", unsafe_allow_html=True)
st.title('Training Dataset Review')
st.caption('Inspect training labels, compare model suggestions, and curate the local trajectory database. '
           'Every label change, review mark and deletion saves immediately.')

DEFAULT_STORE = Path(os.environ.get('MOMA_TRAINSET_STORE',
    str(Path(__file__).resolve().parents[3] / 'datasets/store/lineage.sqlite3')))

# Register the binding alongside its visible label, so they cannot drift apart.
review_shortcuts = {'ArrowLeft': '← Previous', 'ArrowRight': 'Next →'}


def shortcut_button(label, shortcut, *, container=st, **kwargs):
    display_key = shortcut if shortcut == 'Delete' else shortcut.upper()
    display_label = f'{label} [{display_key}]'
    review_shortcuts[shortcut] = display_label
    return container.button(display_label, **kwargs)


def navigate_to(trajectory_key):
    st.session_state['review_key'] = trajectory_key


def navigate_from_control(control_key, trajectory_keys):
    value = st.session_state[control_key]
    if control_key == 'review_key':
        navigate_to(value)
    else:
        position = max(1, min(int(value), len(trajectory_keys)))
        navigate_to(trajectory_keys[position - 1])


@st.cache_data(show_spinner=False)
def catalog(path: str, token: tuple):
    return store.list_versions(Path(path))


@st.cache_data(show_spinner=False)
def trajectory_list(path: str, lineage: str, token: tuple):
    return store.list_trajectories(Path(path), lineage)


@st.cache_data(show_spinner=False)
def recent_history(path: str, token: tuple):
    return pd.DataFrame(store.read_history(Path(path), limit=25))


def cached_queue(path: str, lineage: str):
    revision = store.revision(path)
    cache = st.session_state.get('review_queue_cache')
    if (cache and cache.get('path') == path and cache.get('lineage') == lineage
            and cache.get('revision') == revision):
        return cache['queue'].copy(), revision
    token = signature(Path(path))
    queue = trajectory_list(path, lineage, token).copy()
    st.session_state['review_queue_cache'] = {
        'path': path, 'lineage': lineage, 'revision': revision, 'queue': queue.copy()}
    return queue, revision


def update_queue_cache(result, trajectory_key, reviewed=None, remove=False):
    cache = st.session_state.get('review_queue_cache')
    if not cache or not isinstance(result, dict):
        return
    queue = cache['queue']
    mask = queue.trajectory_key.eq(trajectory_key)
    if remove:
        queue = queue.loc[~mask].copy()
    elif mask.any():
        queue = queue.copy()
        if 'division_points' in result:
            queue.loc[mask, 'division_points'] = int(result['division_points'])
        if 'excluded_points' in result:
            queue.loc[mask, 'excluded_points'] = int(result['excluded_points'])
        if reviewed is not None:
            queue.loc[mask, 'reviewed'] = bool(reviewed)
    cache['queue'] = queue
    cache['revision'] = int(result.get('revision', cache['revision']))


def signature(path: Path) -> tuple:
    stat = path.stat()
    wal = Path(str(path) + '-wal')
    return (stat.st_mtime_ns, stat.st_size, wal.stat().st_mtime_ns if wal.exists() else 0,
            store.revision(path))


def invalidate_view(message: str) -> None:
    st.session_state['review_flash'] = message
    # Predictions depend on measurements, not labels, so keep them across label
    # edits. Evaluation metrics and the pre-edit overlay are label-dependent.
    st.session_state.pop('review_evaluation', None)
    st.session_state.pop('review_view_revision', None)
    st.session_state['review_plot_epoch'] = st.session_state.get('review_plot_epoch', 0) + 1


def mutation_callback(operation, *args, **kwargs) -> None:
    """Save in a widget callback: one rerun, no second full-page refresh."""
    try:
        result = operation(*args, **kwargs)
    except (ValueError, RuntimeError, KeyError, OSError) as error:
        st.session_state['review_error'] = str(error)
        return
    trajectory_key = args[1] if len(args) > 1 else kwargs.get('trajectory_key')
    reviewed = args[2] if operation is store.mark_reviewed else None
    update_queue_cache(result, trajectory_key, reviewed=reviewed)
    invalidate_view('Labels saved' if operation is store.save_labels else 'Saved')


def label_callback(db_path, trajectory_key, frame, observation_ids, action, expected_revision):
    selected = st.session_state.get('review_selected_ids')
    if selected:
        observation_ids = [int(value) for value in selected]
    if not observation_ids:
        st.session_state['review_error'] = 'Select points on the plot first.'
        return
    try:
        changed = review.edit_labels(frame, observation_ids, action)
    except (ValueError, RuntimeError) as error:
        st.session_state['review_error'] = str(error)
        return
    mutation_callback(store.save_labels, db_path, trajectory_key, changed,
                      expected_revision=expected_revision)


def resolve_callback(db_path, trajectory_key, frame, expected_revision):
    try:
        resolved = review.resolve_labels(frame)
    except (ValueError, RuntimeError) as error:
        st.session_state['review_error'] = str(error)
        return
    mutation_callback(store.save_labels, db_path, trajectory_key, resolved['frame'],
                      expected_revision=expected_revision)


def apply_suggestions_callback(db_path, trajectory_key, frame, predictions,
                               apply_division, apply_exclusion, expected_revision):
    changed = frame.copy()
    prediction_by_id = predictions.set_index('fov_cell_index')
    for label, enabled in [('division_point', apply_division),
                           ('excluded_point', apply_exclusion)]:
        if enabled:
            changed[label] = changed.fov_cell_index.map(
                prediction_by_id[f'pred_{label}']).astype(bool)
    mutation_callback(store.save_labels, db_path, trajectory_key, changed,
                      expected_revision=expected_revision)


def refresh(message: str) -> None:
    invalidate_view(message)
    st.rerun()


def remove_current_trajectory(db_path, trajectory_key, expected_revision, next_key):
    # Capture the displayed key/revision in the button callback. The backend
    # rejects stale clicks before writing; the ensuing page run draws a fresh queue.
    try:
        removed = store.delete_trajectory(db_path, trajectory_key,
                                           expected_revision=expected_revision,
                                           confirmation_key=trajectory_key)
    except (ValueError, RuntimeError, KeyError, OSError) as error:
        st.session_state['review_error'] = str(error)
    else:
        if next_key is not None:
            st.session_state['review_pending_key'] = next_key
        update_queue_cache(removed, trajectory_key, remove=True)
        invalidate_view(f'Trajectory {trajectory_key} removed')


def call_mutation(operation, *args, **kwargs):
    try:
        result = operation(*args, **kwargs)
        trajectory_key = args[1] if len(args) > 1 else kwargs.get('trajectory_key')
        update_queue_cache(result, trajectory_key)
        refresh('Labels saved')
    except (ValueError, RuntimeError, KeyError, OSError) as error:
        st.error(str(error))
        st.info('Reload the dataset before retrying if another session changed the database.')


with st.sidebar:
    st.header('Dataset')
    with st.form('open_database'):
        path_text = st.text_input('Training database', value=str(DEFAULT_STORE))
        open_db = st.form_submit_button('Open database', width='stretch')
    if open_db:
        candidate = Path(path_text).expanduser().resolve()
        if not candidate.is_file():
            st.error('Choose an existing SQLite training database.')
        else:
            st.session_state['review_database'] = str(candidate)
            for field in ('review_dataset', 'review_key', 'review_predictions', 'review_evaluation', 'review_original'):
                st.session_state.pop(field, None)
            st.rerun()

path = Path(st.session_state.get('review_database', str(DEFAULT_STORE))).expanduser().resolve()
if not path.is_file():
    st.info('Open a training database in the sidebar to begin.')
    st.stop()
try:
    token = signature(path)
    versions = catalog(str(path), token)
except Exception as error:
    st.error(f'Cannot open the training database: {error}')
    st.stop()
if not versions:
    st.info('This database has no dataset versions.')
    st.stop()

version_by_id = {v['lineage_id']: v for v in versions}
lineages = list(version_by_id)
preferred = next((s for s in reversed(lineages) if 'merged' in s), lineages[-1])
pending_lineage = st.session_state.pop('review_pending_lineage', None)
if pending_lineage in lineages:
    st.session_state['review_dataset'] = pending_lineage
if st.session_state.get('review_dataset') not in lineages:
    st.session_state['review_dataset'] = preferred
with st.sidebar:
    lineage = st.selectbox('Dataset version', lineages, key='review_dataset')
    selected_version = version_by_id[lineage]
    st.caption('Frozen training version' if selected_version['frozen'] else 'Working data — finalize before training')
    st.caption(str(path))
    st.button('Reload database', on_click=lambda: st.cache_data.clear(), width='stretch')
try:
    queue = trajectory_list(str(path), lineage, token).copy()
    current_revision = store.revision(path)
except Exception as error:
    st.error(f'Cannot read this dataset: {error}')
    st.stop()

with st.sidebar:
    sources = sorted(queue['source_dataset'].unique()) if len(queue) else []
    source_filter = st.multiselect('Sources', sources, default=[], help='Empty means all sources.')
    status = st.selectbox('Review status', ['All', 'Not reviewed', 'Reviewed'])
    search = st.text_input('Find trajectory', placeholder='FOV, channel, or trajectory key')
    sort = st.selectbox('Queue order', ['Dataset order', 'Most exclusions', 'Most division labels', 'Model disagreement'])
    st.divider()
    st.header('Model preview')
    tags = list(AI_MODEL_REGISTRY)
    model_tag = st.selectbox('Model checkpoint', tags, key='review_model_tag')
    checkpoint, guidance = AI_MODEL_REGISTRY[model_tag]
    st.caption(guidance)
    device = st.selectbox('Inference device', ['cpu', 'auto', 'cuda:0', 'cuda:1'])
    metrics_path = Path(checkpoint).with_name('metrics.json')
    threshold_defaults = (0.5, 0.5)
    if metrics_path.is_file():
        try:
            threshold_values = json.loads(metrics_path.read_text())['thresholds']
            threshold_defaults = (float(threshold_values['division_point']), float(threshold_values['excluded_point']))
        except (KeyError, ValueError, OSError):
            pass
    # Widget keys include the model so each model opens with its calibrated thresholds.
    division_threshold = st.slider('Division threshold', 0.0, 1.0, threshold_defaults[0], 0.01,
                                   key=f'review_div_threshold_{model_tag}')
    exclusion_threshold = st.slider('Exclusion threshold', 0.0, 1.0, threshold_defaults[1], 0.01,
                                    key=f'review_exc_threshold_{model_tag}')
thresholds = (division_threshold, exclusion_threshold)
from moma_lineage_analysis.trajectory_cleaning.dataset_store import sha256_file
checkpoint_hash = sha256_file(Path(checkpoint))

with st.sidebar:
    st.divider()
    st.header('Training version')
    if selected_version['frozen']:
        st.caption('This version is training-ready. Edits you make next become working data.')
    else:
        st.caption('Edits save into the database as working data. Create one training-ready version when you plan to retrain.')
    with st.expander('Create training-ready version'):
        st.caption('R004 omits trajectories with exclusion runs of five or more frames from this version only; it never deletes them from the database.')
        attestation = st.checkbox('These labels are final for training',
                                  key=f'review_attestation_{lineage}')
        if shortcut_button('Make training-ready', 't', disabled=not attestation, width='stretch'):
            dataset_id = f'reviewed_{datetime.now():%Y%m%d_%H%M%S_%f}'
            try:
                created = store.finalize(path, lineage, dataset_id,
                                         expected_revision=current_revision,
                                         verification_basis='user_reviewed_in_trainset_app')
            except Exception as error:
                st.error(f'Could not create a training-ready version: {error}')
            else:
                st.session_state.pop('review_view_revision', None)
                st.session_state['review_flash'] = 'Training-ready version created'
                st.session_state['review_pending_lineage'] = created['lineage_id']
                st.rerun()

@st.fragment
def review_workspace():
    try:
        queue, current_revision = cached_queue(str(path), lineage)
        token = signature(path)
    except Exception as error:
        st.error(f'Cannot read this dataset: {error}')
        return

    if 'trajectory_key' not in queue and 'key' in queue:
        queue = queue.rename(columns={'key': 'trajectory_key'})
    if 'reviewed' not in queue:
        queue['reviewed'] = False
    if 'review_error' in st.session_state:
        st.error(st.session_state.pop('review_error'))
        st.info('Reload the dataset before retrying if another session changed the database.')
    view_identity = (str(path), lineage)
    view = st.session_state.get('review_view_revision')
    if view and view['identity'] == view_identity and view['revision'] != current_revision:
        st.warning('The database changed in another session. Reload to review the current labels before editing.')
        if st.button('Reload changed dataset', type='primary'):
            st.session_state.pop('review_view_revision', None)
            st.session_state.pop('review_predictions', None)
            st.session_state.pop('review_evaluation', None)
            st.session_state.pop('review_original', None)
            st.session_state['review_plot_epoch'] = st.session_state.get('review_plot_epoch', 0) + 1
            st.cache_data.clear()
            st.rerun()
        st.stop()
    st.session_state['review_view_revision'] = {'identity': view_identity, 'revision': current_revision}
    if 'review_flash' in st.session_state:
        st.toast(st.session_state.pop('review_flash'), icon='✅', duration=3)

    columns = st.columns(4)
    columns[0].metric('Trajectories', len(queue))
    columns[1].metric('Observations', int(queue['rows'].sum()) if len(queue) else 0)
    columns[2].metric('Division labels', int(queue['division_points'].sum()) if len(queue) else 0)
    columns[3].metric('Exclusion labels', int(queue['excluded_points'].sum()) if len(queue) else 0)

    filtered = queue
    if source_filter:
        filtered = filtered[filtered.source_dataset.isin(source_filter)]
    if status != 'All':
        filtered = filtered[filtered.reviewed.eq(status == 'Reviewed')]
    if search:
        filtered = filtered[filtered.trajectory_key.str.contains(search, case=False, regex=False)]
    if sort == 'Most exclusions':
        filtered = filtered.sort_values('excluded_points', ascending=False, kind='stable')
    elif sort == 'Most division labels':
        filtered = filtered.sort_values('division_points', ascending=False, kind='stable')
    elif sort == 'Model disagreement':
        evaluated = st.session_state.get('review_evaluation')
        expected_eval = (str(path), lineage, current_revision, checkpoint_hash, device, thresholds)
        if evaluated and evaluated['signature'] == expected_eval:
            scores = evaluated['scores'].copy()
            scores['disagreements'] = scores.division_disagreements + scores.exclusion_disagreements
            filtered = filtered.merge(scores[['trajectory_key', 'disagreements']], on='trajectory_key', how='left')
            filtered = filtered.sort_values('disagreements', ascending=False, kind='stable')
        else:
            st.info('Evaluate the filtered set to order trajectories by model disagreement.')

    review_tab, evaluation_tab = st.tabs(['Review trajectories', 'Evaluate dataset'])

    with review_tab:
        if filtered.empty:
            st.info('No trajectories match these filters.')
        else:
            keys = filtered.trajectory_key.tolist()
            pending_key = st.session_state.pop('review_pending_key', None)
            if pending_key in keys:
                st.session_state['review_key'] = pending_key
            if st.session_state.get('review_key') not in keys:
                st.session_state['review_key'] = keys[0]
            position = keys.index(st.session_state['review_key'])
            # Synchronize both number controls before instantiating their widgets.
            st.session_state['review_position_slider'] = position + 1
            st.session_state['review_position_input'] = position + 1
            previous, selector, following = st.columns([1, 6, 1])
            previous.button('← Previous', disabled=position == 0, width='stretch',
                            on_click=navigate_to, args=(keys[max(position - 1, 0)],))
            key = selector.selectbox('Trajectory', keys, key='review_key')
            following.button('Next →', disabled=position == len(keys) - 1, width='stretch',
                             on_click=navigate_to, args=(keys[min(position + 1, len(keys) - 1)],))
            slider_col, number_col = st.columns([5, 1])
            slider_col.slider('Trajectory position', min_value=1, max_value=max(2, len(keys)),
                              step=1, key='review_position_slider', disabled=len(keys) == 1,
                              on_change=navigate_from_control, args=('review_position_slider', keys),
                              help='Choose a trajectory number in the current filtered queue.')
            number_col.number_input('Trajectory number', min_value=1, max_value=len(keys),
                                    step=1, key='review_position_input', disabled=len(keys) == 1,
                                    on_change=navigate_from_control, args=('review_position_input', keys))
            st.caption(f'{keys.index(key) + 1} / {len(keys)} in this queue')
            snapshot = store.read_snapshot(path, lineage, trajectory_key=key)
            frame = snapshot['frame']
            frame_revision = snapshot['revision']
            if frame_revision != current_revision:
                st.warning('The database changed while this view was loading. Reload before editing.')
                st.stop()
            row = queue.set_index('trajectory_key').loc[key]
            badge = evaluation.split_membership(Path(checkpoint), [key]).get(key, 'unseen')
            st.write(f'**Source:** {row["source_dataset"]} · **Model split:** {badge} · '
                     f'**Review:** {"reviewed" if row["reviewed"] else "not reviewed"}')
            if badge == 'train':
                st.caption('This model was trained on this trajectory; agreement here is a review aid, not held-out performance.')
            prediction_signature = (str(path), lineage, key,
                                    evaluation.cache_key(frame, Path(checkpoint), device=device,
                                                         thresholds=thresholds, include_labels=False))
            cached = st.session_state.get('review_predictions')
            result = cached['result'] if cached and cached['signature'] == prediction_signature else None
            if shortcut_button('Run model preview', 'p', width='content'):
                try:
                    with st.spinner('Predicting this trajectory…'):
                        result = evaluation.evaluate_frame(frame, Path(checkpoint), device=device, thresholds=thresholds)
                    st.session_state['review_predictions'] = {'signature': prediction_signature, 'result': result}
                except Exception as error:
                    st.error(f'Model preview failed: {error}')
            original = st.session_state.get('review_original')
            if not original or original['key'] != key:
                original = {'key': key, 'frame': frame.copy()}
                st.session_state['review_original'] = original
            figure_context = (str(path), lineage, key, result is not None)
            cached_figure = st.session_state.get('review_figure')
            reusable = (cached_figure is not None
                        and cached_figure.get('context') == figure_context)
            figure = plots.build_trajectory_plot(
                frame, predictions=result['predictions'] if result else None,
                thresholds=thresholds, original_labels=original['frame'],
                plot_epoch=st.session_state.get('review_plot_epoch', 0),
                figure=cached_figure['figure'] if reusable else None)
            st.session_state['review_figure'] = {'context': figure_context, 'figure': figure}
            event = st.plotly_chart(figure, width='stretch', on_select='rerun',
                                    selection_mode=('points', 'box', 'lasso'),
                                    key=f'review_plot_{path}_{lineage}_{key}_{int(result is not None)}')
            st.caption('Removes the whole trajectory from the database and every dataset that references it. Saves immediately; no in-app undo.')
            next_key = (keys[position + 1 if position + 1 < len(keys) else position - 1]
                        if len(keys) > 1 else None)
            shortcut_button('Remove trajectory from database', 'Delete', type='primary',
                      key=f'review_remove_{path}_{key}', on_click=remove_current_trajectory,
                      args=(path, key, frame_revision, next_key))
            selected = plots.selected_observation_ids(event, frame.fov_cell_index.tolist())
            id_strings = [str(int(value)) for value in frame.fov_cell_index]
            selection_context = (str(path), lineage, key)
            if st.session_state.get('review_selection_context') != selection_context:
                st.session_state['review_selected_ids'] = []
                st.session_state['review_selection_context'] = selection_context
                st.session_state['review_plot_selection'] = []
            plot_selection = [str(int(v)) for v in selected]
            if plot_selection != st.session_state.get('review_plot_selection', []):
                st.session_state['review_selected_ids'] = plot_selection
                st.session_state['review_plot_selection'] = plot_selection
            # Selection comes from the plot; no separate ID control is shown.
            explicit_strings = st.session_state.get('review_selected_ids', [])
            explicit = [int(value) for value in explicit_strings]
            st.caption('Select points on the plot, then use the buttons or keys. '
                       'Each button shows its shortcut in brackets; ←/→ navigate. '
                       'Shortcuts pause while typing or adjusting the slider. Changes save immediately.')
            actions = [('Add division', 'add_division', 'd'), ('Remove division', 'remove_division', 'm'),
                       ('Exclude points', 'exclude', 'e'), ('Restore points', 'restore', 'r'),
                       ('Clear labels', 'clear', 'c')]
            action_columns = st.columns(len(actions))
            for col, (label, action, shortcut) in zip(action_columns, actions):
                shortcut_button(label, shortcut, container=col, disabled=not explicit, width='stretch',
                                key=f'review_action_{path}_{key}_{action}', on_click=label_callback,
                                args=(path, key, frame, explicit, action, frame_revision))
            tools = st.columns(3)
            shortcut_button('Resolve division/exclusion conflicts', 'f', container=tools[0],
                            width='stretch', key=f'review_resolve_{path}_{key}',
                            on_click=resolve_callback, args=(path, key, frame, frame_revision))
            shortcut_button('Undo last label edit', 'u', container=tools[1], width='stretch',
                            key=f'review_undo_{path}_{key}', on_click=mutation_callback,
                            args=(store.undo_labels, path, key), kwargs={'expected_revision': frame_revision})
            shortcut_button('Mark not reviewed' if row['reviewed'] else 'Mark reviewed', 'k',
                            container=tools[2], width='stretch', key=f'review_mark_{path}_{key}',
                            on_click=mutation_callback,
                            args=(store.mark_reviewed, path, key, not bool(row['reviewed'])),
                            kwargs={'expected_revision': frame_revision})
            if result:
                with st.expander('Apply model suggestions'):
                    st.caption('Suggestions remain separate from saved labels until you apply them.')
                    apply_division = st.checkbox('Apply division predictions', value=True)
                    apply_exclusion = st.checkbox('Apply exclusion predictions', value=True)
                    shortcut_button('Apply selected suggestion types', 'a',
                                    disabled=not (apply_division or apply_exclusion),
                                    key=f'review_apply_{path}_{key}',
                                    on_click=apply_suggestions_callback,
                                    args=(path, key, frame, result['predictions'],
                                          apply_division, apply_exclusion, frame_revision))
            with st.expander('Edit labels as a table'):
                # The grid is opt-in: it is large and would be rebuilt on every
                # quick plot-label action. Checkbox changes still autosave.
                table_enabled = st.checkbox(
                    'Enable table editor', key=f'review_table_enabled_{path}_{key}')
                if table_enabled:
                    # Keep IDs as decimal strings throughout the browser round trip.
                    label_table = frame[['fov_cell_index', 'time_h', 'spine_length',
                                         'division_point', 'excluded_point']].copy()
                    label_table['fov_cell_index'] = id_strings
                    edited = st.data_editor(label_table,
                                            disabled=['fov_cell_index', 'time_h', 'spine_length'],
                                            hide_index=True, width='stretch',
                                            key=f'review_editor_{path}_{key}')
                    labels = ['division_point', 'excluded_point']
                    try:
                        changed = review.table_labels(frame, edited)
                    except (TypeError, ValueError) as error:
                        st.error(str(error))
                    else:
                        if not changed[labels].equals(frame[labels]):
                            call_mutation(store.save_labels, path, key, changed,
                                          expected_revision=frame_revision)

    with evaluation_tab:
        st.write('Compare model predictions with the current saved labels for this filtered queue.')
        st.caption('Training/validation/test badges refer to the model companion split file. Editing labels does not create a new held-out evaluation.')
        if shortcut_button('Evaluate filtered trajectories', 'v', disabled=filtered.empty):
            try:
                with st.spinner(f'Evaluating {len(filtered)} trajectories…'):
                    evaluation_snapshot = store.read_snapshot(path, lineage)
                    selected_frame = evaluation_snapshot['frame']
                    evaluation_revision = evaluation_snapshot['revision']
                    selected_frame = selected_frame[selected_frame.trajectory_key.isin(filtered.trajectory_key)].reset_index(drop=True)
                    evaluated = evaluation.evaluate_frame(selected_frame, Path(checkpoint), device=device, thresholds=thresholds)
                    scores = evaluation.trajectory_scores(evaluated['predictions'], thresholds)
                st.session_state['review_evaluation'] = {
                    'signature': (str(path), lineage, evaluation_revision, checkpoint_hash, device, thresholds),
                    'frame_cache_key': evaluation.cache_key(selected_frame, Path(checkpoint), device=device, thresholds=thresholds),
                    'result': evaluated, 'scores': scores}
            except Exception as error:
                st.error(f'Evaluation failed: {error}')
        evaluated = st.session_state.get('review_evaluation')
        same_scope = evaluated and set(evaluated['scores'].trajectory_key) == set(filtered.trajectory_key)
        if same_scope and evaluated['signature'] == (str(path), lineage, current_revision, checkpoint_hash, device, thresholds):
            metrics = evaluated['result']['metrics']
            a, b, c = st.columns(3)
            a.metric('Division-event F1', f'{metrics["division_event"]["f1"]:.4f}')
            b.metric('Exclusion F1', f'{metrics["point"]["excluded_point"]["f1"]:.4f}')
            c.metric('Exclusion AP', f'{metrics["point"]["excluded_point"]["average_precision"]:.4f}')
            score_table = evaluated['scores'].copy()
            score_table['model_split'] = score_table.trajectory_key.map(evaluation.split_membership(Path(checkpoint), score_table.trajectory_key.tolist()))
            st.dataframe(score_table, hide_index=True, width='stretch')
            st.download_button('Download evaluation table', score_table.to_csv(index=False),
                               file_name='training_dataset_evaluation.csv', mime='text/csv')
            with st.expander('Full evaluation metrics'):
                st.json(metrics)
        else:
            st.info('Run evaluation to see metrics and a trajectory disagreement table.')

    with review_tab:
        with st.expander('Recent database changes'):
            history = recent_history(str(path), token)
            st.dataframe(history, hide_index=True, width='stretch')



review_workspace()

# Same keydown bridge as the main lineage app. Arrow keys navigate; the letter
# keys trigger the visible label buttons. Inputs, sliders and dialogs keep
# their own keys.
components.html(shortcut_script(review_shortcuts), height=0)