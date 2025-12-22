from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_notebook(notebook_path: Path, output_path: Path, timeout: int = 900) -> None:
    """Execute a notebook and write the executed notebook to output_path."""
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    # Execute in the notebook's folder so relative paths behave as expected
    ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})

    with output_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Oil–Energy project pipeline.")
    parser.add_argument(
        "--notebook",
        type=str,
        default="08_final_economic_analysis.ipynb",
        help="Notebook filename inside ./notebooks to execute.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Execution timeout in seconds.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    notebooks_dir = project_root / "notebooks"
    outputs_dir = project_root / "outputs"

    nb_in = notebooks_dir / args.notebook
    nb_out = outputs_dir / f"executed_{args.notebook}"

    try:
        print(f"[INFO] Executing notebook: {nb_in}")
        run_notebook(nb_in, nb_out, timeout=args.timeout)
        print(f"[INFO] Done. Executed notebook saved to: {nb_out}")
        return 0
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
