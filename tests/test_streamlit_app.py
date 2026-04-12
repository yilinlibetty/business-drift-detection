from __future__ import annotations

import py_compile
from pathlib import Path


def test_streamlit_app_compiles_without_importing_streamlit():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    py_compile.compile(str(app_path), doraise=True)


def test_export_figures_script_compiles():
    script_path = Path(__file__).resolve().parents[1] / "export_figures.py"
    py_compile.compile(str(script_path), doraise=True)
