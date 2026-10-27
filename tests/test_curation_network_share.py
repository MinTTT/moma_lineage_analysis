"""Optional real-SMB integration: set MOMA_TEST_NETWORK_DIR to a writable share."""

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

import pandas as pd
import pytest

from moma_lineage_analysis import curation_store as store
from test_curation_store import session


@pytest.mark.skipif(not os.environ.get("MOMA_TEST_NETWORK_DIR"), reason="No writable network test directory configured")
def test_mapped_share_and_unc_alias_create_save_recover_and_crash_rollback():
    root = Path(os.environ["MOMA_TEST_NETWORK_DIR"])
    assert store._is_network_path(root)
    with tempfile.TemporaryDirectory(prefix="moma-network-sqlite-test-", dir=root) as temporary:
        directory = Path(temporary)
        trajectories, labels, key = session(directory / "input")
        db = directory / store.DB_FILENAME
        unc_db = db.resolve()
        sources = store.fingerprint_sources([trajectories[key]["source_csv_path"]])
        info = store.create_dataset(
            db, trajectory_dict=trajectories, labeled_frames=labels,
            selected_keys={key}, sources=sources, settings={}, logic_version="2.3", current_key=key,
        )
        assert info["created"]
        assert store._is_network_path(unc_db)
        # Reproduce the original authority bug only against our temporary DB.
        try:
            with closing(sqlite3.connect(unc_db.as_uri() + "?mode=ro", uri=True)):
                old_uri_result = "accepted by this SQLite build"
        except sqlite3.OperationalError as exc:
            old_uri_result = str(exc)
        recovered = store.load_dataset(unc_db, info["dataset_id"])
        entry = recovered["trajectory_dict"][key]
        pd.testing.assert_frame_equal(entry["trajectory"], trajectories[key]["trajectory"].reset_index(drop=True))
        entry["_review_status"] = "reviewed"
        entry["_reviewed_at"] = "2026-10-27T00:00:00+00:00"
        assert store.save_changes(
            db, info["dataset_id"], expected_revision=0,
            trajectory_dict={key:entry}, labeled_frames=labels,
            selected_keys={key}, current_key=key,
        ) == 1
        assert store.load_dataset(unc_db, 1)["trajectory_dict"][key]["_review_status"] == "reviewed"
        with store._connect(unc_db, write=True) as connection:
            assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
            assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        with pytest.raises(store.CurationConflictError):
            store.save_changes(unc_db, 1, expected_revision=0, trajectory_dict={},
                               labeled_frames={}, selected_keys=set(), current_key=key)
        # A killed remote writer must not leak a partial label update.
        program = '''
import os,sys
from moma_lineage_analysis import curation_store as store
with store._connect(sys.argv[1], write=True) as connection:
    connection.execute("UPDATE observations SET _moma_excluded_point=1")
    os._exit(73)
'''
        killed = subprocess.run([sys.executable, "-c", program, str(unc_db)], timeout=30)
        assert killed.returncode == 73
        after_crash = store.load_dataset(db, 1)
        assert after_crash["revision"] == 1
        assert after_crash["trajectory_dict"][key]["excluded_indices"] == {20}
        with closing(sqlite3.connect(str(db))) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert not connection.execute("PRAGMA foreign_key_check").fetchall()
            assert connection.execute("SELECT sum(excluded_point) FROM selected_labeled").fetchone()[0] == 1
        assert not db.with_name(db.name + "-wal").exists()
        assert not db.with_name(db.name + "-shm").exists()
        print(json.dumps({"mapped_drive":root.drive, "unc_alias":unc_db.anchor,
                          "original_uri_error":old_uri_result, "journal":"DELETE", "synchronous":"FULL",
                          "create_save_unc_recovery":"pass", "revision_conflict":"pass",
                          "killed_writer_rollback":"pass", "integrity_check":"ok"}))
