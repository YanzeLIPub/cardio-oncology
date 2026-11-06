"""Configurable local paths for the source-only revision release."""
from pathlib import Path
import os
import re

LAYOUT = {"project": "project", "china": "historical_china", "pipeline": "historical_pipeline", "njm": "controlled_data/njm", "mimic": "controlled_data/mimic", "workspace": "historical_workspace"}

def work_root():
    return Path(os.environ.get("CARDIO_WORK_ROOT", str(Path.home() / "cardio_revision_work"))).expanduser().resolve()

def root(name):
    default = work_root() / LAYOUT[name]
    return Path(os.environ.get("CARDIO_" + name.upper() + "_ROOT", str(default))).expanduser().resolve()

def path(name, relative=""):
    # Source-file dependencies use the copied release code, even when input data
    # are supplied in an existing private modelling directory.
    staged = work_root() / LAYOUT[name] / relative
    if Path(relative).suffix in {".py", ".pyi"} and staged.is_file():
        result = staged
    else:
        result = root(name) / relative
    text = result.as_posix()
    return text + "/" if relative.endswith("/") and not text.endswith("/") else text

def source_path(value):
    value = Path(value)
    for name, folder in LAYOUT.items():
        try:
            relative = value.relative_to(root(name))
        except ValueError:
            continue
        staged = work_root() / folder / relative
        if staged.is_file():
            return staged
    return value

def font_path(filename):
    directory = os.environ.get("CARDIO_FONT_DIR")
    if directory:
        return str(Path(directory).expanduser() / filename)
    windows = os.environ.get("WINDIR")
    if windows:
        return str(Path(windows) / "Fonts" / filename)
    from matplotlib.font_manager import FontProperties, findfont
    weight = "bold" if filename.lower().endswith("bd.ttf") else "normal"
    return findfont(FontProperties(family="Arial", weight=weight), fallback_to_default=False)

def resolve_config(value):
    if isinstance(value, dict):
        return {k: resolve_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_config(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"\$\{CARDIO_(PROJECT|CHINA|PIPELINE|NJM|MIMIC|WORKSPACE)_ROOT\}", lambda m: path(m.group(1).lower()), value)
    return value
