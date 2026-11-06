"""Visualize the six source tables and compose vector panels with the retained figures."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import sys, json, shutil
R = Path(__file__).resolve().parents[1]
OUT = R / 'figures'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import TwoSlopeNorm
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.2, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': 0.55, 'xtick.major.width': 0.55, 'ytick.major.width': 0.55})
BLUE = '#27689b'
ORANGE = '#bf6938'
INK = '#202a32'
GREY = '#b5bcc3'
SAVED = {}

def figure(w, h):
    fig = plt.figure(figsize=(w / 72, h / 72))
    fig.patch.set_facecolor('white')
    return fig

def heading(fig, letter, title):
    labels = {'Original XGBoost parameters': 'a', 'Forest parameters and ensemble': 'b', 'Original fitting and evaluation paths': 'c'}
    letter = labels.get(title, '')
    if letter:
        fig.text(0.012, 0.965, letter + ')', fontsize=8.5, fontweight='bold', va='top')
    fig.text(max(0.055, 18 / (fig.get_figwidth() * 72)) if letter else 0.012, 0.965, title, fontsize=8.0, fontweight='bold', va='top')

def save(fig, key, w, h):
    fig.savefig(OUT / f'Figure_{key}.pdf', facecolor='white')
    fig.savefig(OUT / f'Figure_{key}.png', dpi=360, facecolor='white')
    plt.close(fig)
    SAVED[key] = {'width': w, 'height': h}

def setting_tracks(key, letter, title, rows, color, forest=False):
    w, h = (234, 148)
    fig = figure(w, h)
    heading(fig, letter, title)
    fig.text(0.055, 0.84, 'Shared values: China and MIMIC', fontsize=6.5, color='#53616c')
    for i, (label, value, upper) in enumerate(rows):
        top = 40 + i * 24
        ax = fig.add_axes([0.39, 1 - (top + 9) / h, 0.51, 13 / h])
        ax.set_xlim(0, upper)
        ax.set_ylim(-1, 1)
        ax.plot([0, upper], [0, 0], color=GREY, lw=0.7)
        ax.plot(value, 0, 'o', color=color, ms=4)
        ax.text(value, 0.43, f'{value:g}', ha='center', fontsize=7.2, fontweight='bold', color=color)
        ax.set_xticks([0, upper])
        ax.set_xticklabels([0, f'{upper:g}'], fontsize=6.0)
        ax.tick_params(axis='x', length=2, pad=1)
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.text(0.025, 1 - (top + 1) / h, label, fontsize=7, va='center')
    if forest:
        ax = fig.add_axes([0.025, 0.07, 0.95, 0.14])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.add_patch(Rectangle((0, 0), 0.5, 0.7, color=BLUE))
        ax.add_patch(Rectangle((0.5, 0), 0.5, 0.7, color=ORANGE))
        ax.text(0.25, 0.35, '0.5 × XGBoost', ha='center', va='center', color='white', fontsize=7)
        ax.text(0.75, 0.35, '0.5 × forest', ha='center', va='center', color='white', fontsize=7)
        fig.text(0.025, 0.235, 'Ensemble probability weights', fontsize=6.5)
    else:
        fig.text(0.025, 0.025, 'Each track has its own parameter scale.', fontsize=6.2, color='#53616c')
    save(fig, key, w, h)

def workflow():
    w, h = (468, 126)
    fig = figure(w, h)
    heading(fig, 'h', 'Original fitting and evaluation paths')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')
    rows = [('China CV', ['Patient means\n104 inputs; zero fill', 'Stratified 5-fold\npatient rows', 'Fit ≤300 rounds\nouter stop: 20', 'Outer validation\npredictions']), ('MIMIC CV', ['Admission rows\n43 input slots', 'Stratified 5-fold\nfit-only median', 'Fit ≤300 rounds\nouter stop: 20', 'Outer validation\n26 effective inputs']), ('China transfer', ['Patient means\n104 inputs; zero fill', 'Whole source\nsource label weights', 'Fit 200 rounds\nno stopping', 'Other Chinese\ncentre']), ('MIMIC transfer', ['Common inputs\nChinese medians', 'Whole source\nsource-only median', 'Fit 300 rounds\nno stopping', 'Target-centre\npredictions'])]
    xs = [70, 169, 268, 367]
    width = 90
    for i, (label, steps) in enumerate(rows):
        y = 23 + i * 23
        ax.text(4, y + 9, label, fontsize=6.5, va='center', fontweight='bold')
        for j, step in enumerate(steps):
            x = xs[j]
            fill = '#e9f1f7' if i < 2 else '#f5eee7'
            ax.add_patch(Rectangle((x, y), width, 19, facecolor=fill, edgecolor=BLUE if i < 2 else ORANGE, lw=0.55))
            ax.text(x + width / 2, y + 9.5, step, ha='center', va='center', fontsize=6.3, linespacing=1.13)
            if j < 3:
                ax.add_patch(FancyArrowPatch((x + width + 1, y + 9.5), (xs[j + 1] - 1, y + 9.5), arrowstyle='-|>', mutation_scale=5, lw=0.65, color=INK))
    fig.text(0.012, 0.012, 'Original CV used outer-validation labels for stopping; patient-isolated settings are in Table 2.', fontsize=6.2, color='#53616c')
    save(fig, '1h', w, h)

def conversion():
    w, h = (468, 132)
    fig = figure(w, h)
    heading(fig, 'c', 'Unit-conversion factors for the four shared laboratory measures')
    ax = fig.add_axes([0.35, 0.18, 0.4, 0.59])
    ys = np.arange(4)[::-1]
    vals = [10, 10, 1, 10]
    for y, v in zip(ys, vals):
        color = BLUE if v == 10 else ORANGE
        ax.plot([0, v], [y, y], color=color, lw=3, solid_capstyle='round')
        ax.scatter([v], [y], s=18, color=color, zorder=3)
        ax.text(v + 0.25, y, '×' + str(v), va='center', fontsize=7, color=color, fontweight='bold')
    ax.set_xlim(0, 11.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([])
    ax.set_xticks([0, 1, 5, 10])
    ax.tick_params(axis='x', labelsize=6.5, length=2, pad=1)
    ax.set_xlabel('Multiplier applied to MIMIC value', fontsize=6.8, labelpad=2)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    names = ['Total protein', 'Albumin', 'CBC hematocrit', 'CBC hemoglobin']
    ids = [50976, 50862, 51221, 51222]
    for i, (name, item) in enumerate(zip(names, ids)):
        fy = 0.18 + 0.59 * (3.5 - i) / 4
        fig.text(0.02, fy + 0.009, name, va='center', fontsize=7.4, fontweight='bold')
        fig.text(0.02, fy - 0.047, f'MIMIC ITEMID {item}', va='center', fontsize=6.1, color='#53616c')
        fig.text(0.8, fy, 'g/dL → g/L' if i != 2 else '% → %', va='center', fontsize=7.2)
    fig.text(0.8, 0.82, 'MIMIC → WHU', fontsize=6.5, color='#53616c')
    save(fig, '2c', w, h)

def versions():
    w, h = (468, 158)
    fig = figure(w, h)
    heading(fig, 'f', 'Recovered historical software versions')
    ax = fig.add_axes([0.22, 0.17, 0.76, 0.64])
    ax.set_xlim(-0.45, 3.7)
    ax.set_ylim(11.55, -0.55)
    labels = ['Python', 'XGBoost', 'NumPy', 'SciPy', 'scikit-learn', 'pandas', 'LightGBM', 'lifelines', 'Matplotlib', 'umap-learn', 'pynndescent', 'SHAP']
    observed = {0: {0: '3.14.2', 1: '3.2.0', 2: '2.3.5', 3: '1.17.0'}, 1: {0: '3.14.2', 1: '3.2.0', 2: '2.4.2', 3: '1.17.1', 4: '1.8.0', 5: '2.3.3', 6: '4.6.0', 7: '0.30.1', 8: '3.10.8'}, 2: {9: '0.5.11', 10: '0.6.0'}, 3: {8: '3.10.8'}}
    for y in range(12):
        ax.axhline(y, color='#eef0f2', lw=0.5, zorder=0)
    for x in range(4):
        for y in range(12):
            if y in observed.get(x, {}):
                ax.scatter(x - 0.29, y, s=13, color=BLUE, zorder=3)
                ax.text(x - 0.15, y, observed[x][y], fontsize=6.7, va='center')
            elif (x, y) == (0, 7):
                ax.scatter(x - 0.29, y, s=16, color=ORANGE, marker='D', zorder=3)
                ax.text(x - 0.15, y, '0.30.1', fontsize=6.7, va='center')
            elif (x, y) in [(0, 4), (2, 11)]:
                ax.scatter(x - 0.29, y, s=16, color='#8a3d40', marker='x', linewidths=0.9, zorder=3)
            else:
                ax.scatter(x - 0.29, y, s=8, facecolors='white', edgecolors='#c6ccd1', linewidths=0.5)
    ax.set_yticks(range(12))
    ax.set_yticklabels(labels, fontsize=6.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['China', 'MIMIC v13', 'Interpretation', 'Figure metadata'], fontsize=7)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.text(0.012, 0.09, '● Observed version', fontsize=6.5, color=BLUE)
    fig.text(0.285, 0.09, '◆ Separate tool query', fontsize=6.5, color=ORANGE)
    fig.text(0.59, 0.09, '× Unresolved version', fontsize=6.5, color='#8a3d40')
    fig.text(0.012, 0.018, 'Open circles: no version observation assigned here; they do not indicate that a package was absent.', fontsize=6.2, color='#53616c')
    save(fig, '3f', w, h)

def candidates():
    w, h = (468, 136)
    fig = figure(w, h)
    heading(fig, 'j', 'Parameter space of the four nested-search candidates')
    values = np.array([[5, 0.05, 1, 1], [3, 0.05, 1, 1], [5, 0.05, 5, 5], [5, 0.025, 1, 1]], float)
    relative = np.log2(values / values[0])
    ax = fig.add_axes([0.21, 0.18, 0.62, 0.58])
    im = ax.imshow(relative, cmap='PuOr_r', norm=TwoSlopeNorm(vmin=-np.log2(5), vcenter=0, vmax=np.log2(5)), aspect='auto')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{values[i, j]:g}', ha='center', va='center', fontsize=7.8, color='white' if relative[i, j] > 2 else INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Depth', 'Learning rate', 'Min child weight', 'L2 lambda'], fontsize=6.7)
    ax.xaxis.tick_top()
    ax.set_yticks(range(4))
    ax.set_yticklabels(['Fixed', 'Shallow', 'Regularized', 'Lower rate'], fontsize=7)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.3)
    ax.tick_params(which='both', length=0, pad=4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = fig.add_axes([0.88, 0.18, 0.023, 0.58])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks(np.log2([0.2, 0.5, 1, 2, 5]), labels=['0.2×', '0.5×', '1×', '2×', '5×'])
    cb.ax.tick_params(labelsize=6.2, length=2, pad=2)
    cb.outline.set_linewidth(0.4)
    fig.text(0.012, 0.035, 'Cell labels: actual parameter values. Colour: ratio to the fixed candidate, on a logarithmic scale.', fontsize=6.3, color='#53616c')
    save(fig, '4j', w, h)
    alt = figure(468, 200)
    ax = alt.add_axes([0.03, 0.08, 0.78, 0.85], projection='3d')
    ax.scatter(values[:, 0], values[:, 1], values[:, 2], c=values[:, 3], cmap='viridis', s=40, vmin=1, vmax=5)
    for row, label in zip(values, ['Fixed', 'Shallow', 'Regularized', 'Lower rate']):
        ax.text(*row[:3], label, fontsize=7)
    ax.set_xlabel('Depth', fontsize=7, labelpad=0)
    ax.set_ylabel('Learning rate', fontsize=7, labelpad=0)
    ax.set_zlabel('Min child weight', fontsize=7, labelpad=0)
    ax.tick_params(labelsize=6, pad=0)
    ax.view_init(elev=24, azim=-56)
    alt.text(0.78, 0.8, 'Colour = L2 lambda\n1 to 5', fontsize=7)
    alt.savefig(R / 'review/candidates_3d_comparison.png', dpi=240)
    plt.close(alt)

def compose():
    import pymupdf as f
    sys.path.pop(0)
    src = R.parent / 'reviewer_response_locations_v1/submission_package'
    metadata = []
    for i in range(1, 6):
        shutil.copy2(src / f'upload/Figure_{i}.pdf', OUT / f'Figure_{i}.pdf')
        shutil.copy2(src / f'author_reference/main_figures/Figure_{i}.png', OUT / f'Figure_{i}.png')
    layouts = {26: [('1f', 0, 0), ('1g', 234, 0), ('1h', 0, 154)], 27: [('4j', 0, 0)], 28: [('3f', 0, 0)], 29: [('2c', 0, 0)]}
    for i, items in layouts.items():
        height = max((y + SAVED[key]['height'] for key, x, y in items))
        doc = f.open()
        pg = doc.new_page(width=468, height=height)
        boxes = {}
        for key, x, y in items:
            panel = f.open(OUT / f'Figure_{key}.pdf')
            box = f.Rect(x, y, x + SAVED[key]['width'], y + SAVED[key]['height'])
            pg.show_pdf_page(box, panel, 0)
            boxes[key] = list(box)
        doc.save(OUT / f'Supplementary_Figure_S{i}.pdf', garbage=4, deflate=True)
        pg.get_pixmap(matrix=f.Matrix(4, 4)).save(OUT / f'Supplementary_Figure_S{i}.png')
        metadata.append({'figure': f'S{i}', 'width_pt': 468, 'height_pt': pg.rect.height, 'panels': boxes})
    (R / 'review/figure_composition.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
if __name__ == '__main__':
    setting_tracks('1f', 'f', 'Original XGBoost parameters', [('Depth', 5, 10), ('Learning rate', 0.05, 0.1), ('Row subsample', 0.8, 1), ('Column subsample', 0.8, 1)], BLUE)
    setting_tracks('1g', 'g', 'Forest parameters and ensemble', [('Trees', 300, 500), ('Depth', 8, 10), ('Min leaf size', 5, 10)], ORANGE, True)
    workflow()
    conversion()
    versions()
    candidates()
    compose()
    (R / 'review/plot_environment.json').write_text(json.dumps({'python': sys.version.split()[0], 'matplotlib': matplotlib.__version__, 'numpy': np.__version__, 'source_tables': 'reviewer_response_locations_v1/Manuscript_revision.docx: Tables 2,3,4,6,7,8', 'scientific_analysis_rerun': False}, indent=2), encoding='utf-8')
    print('Six graphical panels in supplementary Figures S26-S29; five main figures retain their data panels.')
