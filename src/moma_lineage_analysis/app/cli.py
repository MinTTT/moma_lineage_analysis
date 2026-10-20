"""Console entry point: launch the MoMa Lineage Streamlit app.

Installed as the ``moma-lineage`` console script; also reachable via
``python -m moma_lineage_analysis.app``. Extra arguments are forwarded to
``streamlit run``.
"""
import sys
from pathlib import Path


def main() -> int:
    from streamlit.web import cli as stcli

    app_path = Path(__file__).resolve().parent / "Moma_lineage_streamlit.py"
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
