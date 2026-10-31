"""Launch the focused training dataset review app."""
import sys
from pathlib import Path

def main() -> int:
    from streamlit.web import cli as stcli
    app_path = Path(__file__).resolve().parent / "streamlit_app.py"
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    return stcli.main()

if __name__ == "__main__":
    raise SystemExit(main())
