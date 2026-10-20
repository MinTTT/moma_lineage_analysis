"""Allow ``python -m moma_lineage_analysis.app`` to launch the Streamlit app."""
from moma_lineage_analysis.app.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
