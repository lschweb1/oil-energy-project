from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _check_paths() -> None:
    """Basic sanity checks for the expected repository structure."""
    required = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "notebooks",
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "\n".join(f"- {p.relative_to(PROJECT_ROOT)}" for p in missing)
        raise FileNotFoundError(f"Missing expected project files/folders:\n{msg}")


def main() -> None:
    _check_paths()

    print("Oil–Energy Relationship Project")
    print("- Repository structure OK")
    print(f"- Project root: {PROJECT_ROOT}")

    print("\nKey folders:")
    print(f"- data/      : {PROJECT_ROOT / 'data'}")
    print(f"- notebooks/ : {PROJECT_ROOT / 'notebooks'}")
    print(f"- src/       : {PROJECT_ROOT / 'src'}")
    print(f"- results/   : {PROJECT_ROOT / 'results'}")

    figures_dir = PROJECT_ROOT / "results" / "figures"
    if figures_dir.exists():
        n_fig = len(list(figures_dir.glob('**/*.*')))
        print(f"\nFigures: {figures_dir} ({n_fig} files)")
    else:
        print(f"\nFigures: {figures_dir} (not found)")

    print("\nHow to reproduce:")
    print("1) Create environment (see environment.yml or requirements.txt)")
    print("2) Run notebooks in order: 01 -> 08 (see notebooks/)")
    print("3) Outputs are saved under results/ and outputs/")


if __name__ == "__main__":
    main()