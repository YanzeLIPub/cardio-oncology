"""
Lancet Digital Health figure style constants.

Usage:
    from utils.style import apply_lancet_style, LANCET
    apply_lancet_style()
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import matplotlib as mpl
LANCET = {'SCH': '#2166AC', 'WHU': '#D6604D', 'MIMIC': '#1a9850', 'transfer': '#969696', 'Low': '#43A047', 'Medium': '#FB8C00', 'High': '#D32F2F', 'font': 'Arial', 'fontsize_tick': 7, 'fontsize_axis': 8, 'fontsize_panel': 9, 'fontsize_title': 9, 'dpi': 600, 'single_col': 85 / 25.4, 'double_col': 170 / 25.4, 'grid_color': '#E5E5E5', 'grid_linewidth': 0.5}

def apply_lancet_style() -> None:
    """Apply Lancet Digital Health rcParams to all subsequent plots."""
    mpl.rcParams.update({'font.family': ['Arial', 'DejaVu Sans'], 'font.size': LANCET['fontsize_axis'], 'axes.facecolor': 'white', 'figure.facecolor': 'white', 'axes.edgecolor': '#333333', 'axes.linewidth': 0.8, 'axes.grid': False, 'grid.color': LANCET['grid_color'], 'grid.linewidth': LANCET['grid_linewidth'], 'xtick.labelsize': LANCET['fontsize_tick'], 'ytick.labelsize': LANCET['fontsize_tick'], 'legend.fontsize': LANCET['fontsize_tick'], 'legend.framealpha': 0.9, 'legend.edgecolor': '#CCCCCC', 'savefig.dpi': LANCET['dpi'], 'savefig.bbox': 'tight', 'savefig.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False})
