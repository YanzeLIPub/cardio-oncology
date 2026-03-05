"""
Lancet Digital Health figure style constants.

Usage:
    from utils.style import apply_lancet_style, LANCET
    apply_lancet_style()
"""
import matplotlib as mpl

# ── Color palette ─────────────────────────────────────────────
LANCET = {
    # Cohort colours  ── Lancet Digital Health palette ──
    'SCH':      '#2166AC',   # deep blue  (ColorBrewer RdBu)
    'WHU':      '#D6604D',   # muted red-orange
    'MIMIC':    '#1a9850',   # dark green (ColorBrewer RdYlGn)
    'transfer': '#969696',   # grey for cross-cohort transfer
    # Risk group colours (KM tertiles)
    'Low':    '#43A047',     # green
    'Medium': '#FB8C00',     # amber
    'High':   '#D32F2F',     # red
    # Typography
    'font':            'Arial',
    'fontsize_tick':   7,
    'fontsize_axis':   8,
    'fontsize_panel':  9,
    'fontsize_title':  9,
    # Resolution
    'dpi': 600,
    # Figure widths in inches
    'single_col': 85 / 25.4,    # 85 mm
    'double_col': 170 / 25.4,   # 170 mm
    # Grid
    'grid_color': '#E5E5E5',
    'grid_linewidth': 0.5,
}


def apply_lancet_style() -> None:
    """Apply Lancet Digital Health rcParams to all subsequent plots."""
    mpl.rcParams.update({
        # Arial first; DejaVu Sans as fallback to cover any missing glyphs
        'font.family':        ['Arial', 'DejaVu Sans'],
        'font.size':          LANCET['fontsize_axis'],
        'axes.facecolor':     'white',
        'figure.facecolor':   'white',
        'axes.edgecolor':     '#333333',
        'axes.linewidth':     0.8,
        'axes.grid':          False,
        'grid.color':         LANCET['grid_color'],
        'grid.linewidth':     LANCET['grid_linewidth'],
        'xtick.labelsize':    LANCET['fontsize_tick'],
        'ytick.labelsize':    LANCET['fontsize_tick'],
        'legend.fontsize':    LANCET['fontsize_tick'],
        'legend.framealpha':  0.9,
        'legend.edgecolor':   '#CCCCCC',
        'savefig.dpi':        LANCET['dpi'],
        'savefig.bbox':       'tight',
        'savefig.facecolor':  'white',
        'axes.spines.top':    False,
        'axes.spines.right':  False,
    })
