# Training-dataset review app

`moma-trainset-review` is an independent Streamlit application for inspecting
and curating the **local SQLite training dataset store**. It is separate from
the lineage-curation app and is distributed in the package as
`src/moma_lineage_analysis/trainset_review/`.

This guide describes the implemented behaviour. The scientific and data-editing
contract is [the cleaning-methods contract](METHODS_TRAJECTORY_CLEANING_MODEL.md)
(round R018).

## What the app does

- Opens the existing dataset store, catalogues its versions, and shows
  trajectory counts, observations, division labels and exclusion labels.
- Filters by source, review status and trajectory key; sorts by label count or
  model disagreement.
- Plots one trajectory and supports label edits: add/remove division, mark
  exclusion, restore exclusion, clear labels, resolve division/exclusion
  conflicts, and mark reviewed.
- Runs a packaged checkpoint as a **preview**, with calibrated default
  thresholds and an explicit, separate **Apply** action.
- Evaluates a filtered queue against a checkpoint and reports point and
  division-event metrics plus training/validation/test/unseen membership.
- Edits the SQLite store directly with immediate autosave; there is no dataset
  CSV/export tab and no download of the dataset.
- Creates a **new frozen training-ready version** from reviewed working data for
  training, with an R004 membership filter and human-attestation checkbox. This
  is a training input, not a file export.

It does **not** train models, select a model, promote a checkpoint, or silently
change labels. Individual measurement values (`time_h`, `spine_length`) are not
editable in this release; label edits and permanent trajectory deletion are the
review mutations.

## Launch

Use the conda environment that has the editable package and Streamlit:

~~~powershell
$py = 'G:\miniconda3\envs\Moma_cellpose\python.exe'
cd G:\python_code\moma_lineage_analysis
& $py -m moma_lineage_analysis.trainset_review
~~~

The console script is installed in that environment, but its Scripts directory
is **not on PATH** by default. Use the absolute path:

~~~powershell
& 'G:\miniconda3\envs\Moma_cellpose\Scripts\moma-trainset-review.exe'
~~~

Choose a host/port with Streamlit server options, for example when port 8501 is
already in use:

~~~powershell
& 'G:\miniconda3\envs\Moma_cellpose\Scripts\moma-trainset-review.exe' --server.address 127.0.0.1 --server.port 8502
~~~

Arguments after `--` are passed to the app rather than Streamlit.

## Open a database

The default is `datasets/store/lineage.sqlite3` in the source checkout. Override
without editing code:

~~~powershell
$env:MOMA_TRAINSET_STORE = 'D:\data\lineage.sqlite3'
& $py -m moma_lineage_analysis.trainset_review
~~~

You can also type an existing SQLite file under **Training database** and click
**Open database**. Opening, listing versions, reading trajectories and plotting
are read-only: the review extension tables are created only on the first
mutating call, so a read-only open never writes review bookkeeping.

Use a copy for experiments. Do not mutate the repository's live store merely to
exercise the UI.

## Two tabs

The app has exactly two tabs: **Review trajectories** and **Evaluate dataset**.
The review tab contains the plot, label controls, table editor and permanent
deletion. The evaluation tab contains model metrics and a download of the
evaluation table only; neither tab offers a dataset CSV export.

## Review trajectories tab

Choose a **Dataset version**, then filter by source, review status, key search
and queue order. Above the plot, drag **Trajectory position** or enter a
1-based **Trajectory number** to jump within that filtered queue. The slider,
number input, trajectory dropdown and previous/next arrows stay synchronized.
Sorting changes the number associated with a trajectory; its identity stays
selected when it remains in the queue. A single-item queue disables both
number controls, and an empty queue hides them.

The review workspace is a `st.fragment`. Reannotation, deletion, resolve, undo,
apply and reviewed actions therefore rerun only the plot/controls panel; the
sidebar, page shell and queue metrics outside it are not rebuilt. The fragment
reloads the queue and revision itself, so deletion advances to the next
trajectory and the header counts update in place. Label edits update the cached
queue row in place instead of re-running the 1,000+ trajectory aggregate query.
Version counts are advanced by the known label delta instead of rescanning all
observations, reducing a representative save from about 420 ms to about 204 ms.
The plot figure object is reused and only its trajectory/marker traces are
replaced. The model-preview cache is keyed on measurements, not labels, so a
preview stays available after a label edit. The table grid is opt-in; keeping it
collapsed avoids rebuilding a large grid on every quick plot action. The
recent-change table is cached per database revision and shows at most 25 rows.
Streamlit still refreshes the Plotly element inside the fragment when new traces
are sent; that framework cost dominates the remaining response time.

The trajectory page shows its source, model split badge, review status, the
saved-label plot and optional model preview. Select points with the plot
lasso/box tools, then choose:

- **Add division** / **Remove division**;
- **Exclude points** / **Restore points**;
- **Clear labels**.

There is no Save button. Every button action, every per-row checkbox change in
**Edit labels as a table** (enable it with **Enable table editor**), reviewed
marks and permanent deletion write directly
to the SQLite store and autosave immediately. Save feedback is a small
transient toast in the bottom-left (about three seconds) such as **Labels
saved**; it never inserts a banner above the plot, so the figure does not move.
The long backup path is omitted from the toast and remains available in the
**Recent database changes** table. There is no separate observation-ID picker:
label actions use the current plot selection. The table checklist is not a
staging area: changing a checkbox commits that row on the next Streamlit
rerun. Labels are shared observations across every dataset version that
references the trajectory, so editing anywhere marks each referencing version
as working. Do not expect a frozen version to remain byte-identical after
review.

**Resolve division/exclusion conflicts** explicitly applies the packaged
effective-division resolver. Opening or plotting never applies it implicitly.

**Undo last label edit** restores only the preceding label payload for that
trajectory. It is not a general database rollback and cannot restore a deleted
trajectory.

Every label edit, conflict resolution and deletion takes an SQLite online backup
first. Reviewed marks also autosave immediately and are audited, but do not
create a full database snapshot. Snapshots are stored in `review_backups/`
beside the selected database (under its parent directory), with path, sha256,
size and an audit event.

### Keyboard shortcuts

Both this app and the main lineage-curation app use the same shared shortcut
handler. The review bindings are:

| Key | Action |
|---|---|
| `ArrowLeft` / `ArrowRight` | Previous / next trajectory in the filtered queue |
| `D` | Add division to selected points |
| `M` | Remove division from selected points |
| `E` | Exclude selected points |
| `R` | Restore selected points |
| `C` | Clear both labels on selected points |
| `F` | Resolve division/exclusion conflicts |
| `U` | Undo the last label edit |
| `K` | Toggle reviewed for the current trajectory |
| `Delete` | Immediately remove the current trajectory from the database |
| `A` | Apply the checked model suggestion types |
| `T` | Create a training-ready version after attestation |
| `P` | Run model preview |
| `V` | Evaluate the filtered trajectories |

Every dataset-modifying button displays its key in brackets, for example
**Add division [D]**. Letter shortcuts work in lowercase and uppercase. The
main lineage app keeps its original bindings; these additional keys apply to
the training-review app.

A shortcut clicks the corresponding visible button only when it is enabled.
Selection actions require selected observation IDs; **A** requires a current
preview and the model-suggestions expander open. **T** requires the sidebar
training-ready expander open and its attestation checked. The handler ignores
held-key repeats, Ctrl/Alt/Meta combinations and composition events. It does
not fire in a text field, contenteditable region, select, slider, combobox,
listbox, data table/editor or open dialog, or against a hidden tab/closed
expander. Slider arrows and numeric-input keys keep their normal editing behavior.

### Concurrent sessions and stale views

Every mutation uses an expected global revision and strict compare-and-swap. A
stale expected revision is rejected and writes nothing. The page also warns
when the database revision changed underneath a view and asks you to reload.
Reload before retrying; do not bypass the warning.

## Plot style

The review plot uses the same style module as the main lineage app
(`src/moma_lineage_analysis/trajectory_view_style.py`): cell-cycle-coloured
segmented spine runs, logarithmic spine-length axis, shared applied/AI markers,
lasso selection and unified x hover. With model predictions it adds a
probability panel using the same cycle colours and thresholds. Editing a label
rebuilds the plot without changing the measurement axes.

## Model preview and evaluation

The model panel lists the shared packaged registry. The default is the
production **R019 CNN**, sha256
`816a8b392ec206dca578a093c6cbc2fc9ec5ba12480d6aca7c8b08349796af99`
(promoted 2026-10-31 from the reviewed five-batch corpus).
The **R015 CNN**, trained on `verified_merged_20261029@v1`, is selectable:
`9753298022805b4bc9f277d70e4da29aa8d88cac84a9575691477f332ff21687`.
Selecting R015 is not promotion; production remains R019.

Threshold sliders start at checkpoint-calibrated defaults. **Run model preview**
only computes suggestions. **Apply selected suggestion types** is a separate,
explicit save action. Cache identity covers ordered frame content, checkpoint
bytes, device, thresholds and revision, so an edited working frame cannot reuse
a stale prediction.

**Evaluate dataset** compares saved labels with the selected checkpoint for the
filtered queue. The split badge comes from the checkpoint's `splits.json`:
`train`, `validation`, `test`, or `unseen` when the key is absent. Agreement on
a training trajectory is a review aid, not held-out performance. Editing labels
does not create a new held-out evaluation.

Evaluation performs inference and reads metrics; it does not calibrate
thresholds, select a model or write labels. CPU, auto and CUDA devices are
offered. Device choice affects speed; do not treat small cross-device
average-precision differences as model differences.

## Permanent trajectory deletion

The red **Remove trajectory from database** button is immediately below the
plot, above the label actions. One click immediately deletes the
currently displayed trajectory: its observations, every dataset membership,
its undo payloads and its trajectory row. No typing or confirmation dialog is
required. This user-requested workflow replaces the earlier collapsed delete
section and confirmation form.

The change saves immediately and the app selects the next trajectory in the
current queue, or the previous one if the removed trajectory was last. If no
trajectories match the current filters afterward, the app shows an empty queue.
Other trajectories and their labels are preserved. There is no in-app restore.
The pre-delete backup under `review_backups/` is an archival recovery path, not
an in-app undo. Revision checking still rejects a stale click before deleting.

## Create a training-ready version (training only)

Label editing writes to the live SQLite store; it does not produce a dataset
file. When the reviewed labels are ready for training, use the sidebar
**Training version** section and expand **Create training-ready version**.

1. Confirm the labels are final for training and tick **These labels are final
   for training**.
2. Click **Make training-ready**. The app applies the frozen R004 exclusion-run
   rule (omit trajectories whose longest contiguous exclusion run is at least
   five frames) to the membership of the new version only; it never deletes
   those trajectories from the database.
3. The new frozen dataset gets an ID like `reviewed_<timestamp>` (lineage
   `reviewed_<timestamp>@v1`) with `human_verified = 1`,
   `verification_basis = user_reviewed_in_trainset_app`, source-lineage
   provenance and an internal canonical-content pin for store verification. No
   CSV is downloaded or offered as a file; this version exists in the training
   store so training can consume it directly.

Training still requires this frozen training-ready version. Training refuses a
working version because its labels or membership may still be changing. Create
another training-ready version after further edits. The `Download evaluation
table` control in the evaluation tab exports metrics/scores only, never
dataset observations.

~~~powershell
& $py -m moma_lineage_analysis.trajectory_cleaning.cli train `
  --dataset-lineage reviewed_<timestamp>@v1 `
  --store datasets\store\lineage.sqlite3 `
  --require-human-verified --model cnn --output-dir runs\my_reviewed_run
~~~

## Historical CSV import is refused

Once a store has reviewed working data, historical CSV build/import is
deliberately refused. Otherwise re-importing an old corpus could resurrect a
trajectory the user permanently deleted. Create a new training-ready version
from the working store, or build a separate historical store.

## Remove duplicate dataset versions (admin)

The app removes individual trajectories; removing a whole duplicate dataset
version is an admin action outside the UI. Use:

```powershell
python scripts\remove_dataset_versions.py --expected-revision 561 `
  --reason "user: merged corpora duplicate the five batch corpora" `
  --lineage-id verified_merged_20261016@v1 `
  --lineage-id verified_merged_20261017@v1 `
  --lineage-id verified_merged_20261029@v1
```

The backend (`trainset_review.store.remove_dataset_versions`) deletes a version
only when every one of its trajectories is still referenced by another version,
so no observation loses all dataset memberships. It is global-revision
CAS-guarded, takes an online snapshot, and records a `remove_dataset_versions`
audit event. It never deletes observations or trajectory rows. R019 used this
path on 2026-10-31 (3 versions / 3,345 memberships; revision 562) after proving
the merged trajectory sets are subsets of the five batch versions. The
pre-removal database is archived under `store/backups/`; the manifest entries
of the removed corpora carry `store_removed` markers.

## What is and is not preserved

- Before a mutation, the original version pins / counts / frozen flag /
  legacy CSV path and hash are copied into review bookkeeping, and an audit
  event plus online snapshot are recorded.
- The live version's working hash/counts/order are recomputed in place and its
  frozen flag is cleared. Training resolution refuses it until you create a
  training-ready version.
- Snapshots under `review_backups/` retain pre-change database state subject to
  your backup-retention policy; this app does not automatically prune them.
- **Do not claim original frozen lineage bytes remain unchanged after user
  edits.** Original pins and backups are preserved, but the live store is
  intentionally mutable under explicit user authority.

## Failure behaviour

- A failed save leaves the committed revision unchanged and reports the error;
  reload if another session may have changed the database.
- Deleting a trajectory is irreversible from the UI. Use a copy for exploratory
  work.
- The app does not edit the scientific model registry or promote checkpoints.
  R019 production remains default; R015 is selectable only.
- The app does not write to training runs or packaged model artifacts.

## Evidence and verification

R019 used the app to review the five batch corpora (labels autosaved; 45
trajectories permanently deleted), then finalized five training-ready lineages
and retrained the models. Evidence: `runs/store_cleanup_20261031/` and
`runs/reviewed_20261031/`; the review-backend removal tests live in
`tests/trainset_review/test_store.py`.

Implementation and test evidence is in `runs/trainset_review_implementation/`:

- final full suite after labelled shortcuts, trajectory controls and the
  reannotation performance change: **425 passed, 4 skipped, 59 subtests passed**
  in 135.33 s, log `pytest_fast_actions_final.txt`;
- reannotation tests assert that a label action does not request an extra page
  rerun, that the existing figure object has its traces replaced, and that an
  existing model preview survives the edit without inference being repeated;
- generated JavaScript handler executed against 53 behavioral cases covering
  all keys, uppercase letters, repeat/modifier/input guards, disabled/hidden
  buttons, closed expanders, dialogs and replacement of previous listeners;
- trajectory control tests cover synchronized slider/number/dropdown/arrows,
  filters, sorting, lineage changes and removal down to an empty queue;
- one-click removal verified in the browser against an isolated fixture: one
  delete event, pre-delete snapshot and automatic advance to the next trajectory;
  report `one_click_remove_report.json`;
- focused UI-style log `ui_style_tests.txt` passes, covering the shared plot
  style, shortcut guard, no-export UI and autosave behaviour;
- Streamlit AppTest log `ui_app_tests.txt` passes;
- launcher `--help` exits 0 through the absolute environment script path;
- implementation tests used temporary databases only and never wrote the live store;

The original R018 independent verification (task-26) is **PASS with zero open
defects**, including backend, registry, core verification, AppTest, browser
deletion and navigation checks. The subsequent plotting, shortcut, autosave and
no-export refinements were independently verified by task-29 (**PASS**, 14
checks): shared cycle-coloured plot style, arrow/letter shortcuts with input
guards, table-checkbox autosave persisting after reload, absence of dataset
export, sidebar-only training-ready creation, and byte-identical live store/WAL.
The production checkpoint hash remains unchanged, and the review activity
already present in the live store is the user's own and was preserved.
