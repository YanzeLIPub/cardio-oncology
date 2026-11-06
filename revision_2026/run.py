"""Stage source files outside the checkout, then run one analysis entry point."""
from pathlib import Path
import argparse
import os
import runpy
import shutil
import sys
from release_paths import work_root

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Local JSON file containing CARDIO_*_ROOT values")
    parser.add_argument("entry", help="Source path relative to this release, e.g. project/revision_.../run_validation.py")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config:
        import json
        settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
        for key, value in settings.items():
            if key.startswith("CARDIO_") and (key.endswith("_ROOT") or key == "CARDIO_FONT_DIR"):
                os.environ[key] = str(value)
    source = Path(__file__).resolve().parent
    entry = (source / args.entry).resolve()
    if not entry.is_relative_to(source) or entry.suffix != ".py" or not entry.is_file():
        parser.error("Choose an included Python entry point.")
    work = work_root()
    if work == source or work.is_relative_to(source):
        parser.error("CARDIO_WORK_ROOT must be outside the public checkout.")
    for directory in ["project", "historical_china", "historical_pipeline", "historical_workspace"]:
        folder = source / directory
        if folder.exists():
            shutil.copytree(folder, work / directory, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(source / "release_paths.py", work / "release_paths.py")
    staged = work / entry.relative_to(source)
    sys.path[:0] = [str(work), str(staged.parent), str(work / "historical_pipeline"), str(work / "historical_china"), str(work / "project")]
    argv = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [str(staged), *argv]
    runpy.run_path(str(staged), run_name="__main__")

if __name__ == "__main__":
    main()
