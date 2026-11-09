Local supplementary-panel legends and vector recomposition

Entry point: local_legends_supplement.py
Dependency: Python 3 and PyMuPDF (fitz).

Example (from the repository root):
python revision_2026-11-09/main_panels_local_legends_v1/figure_work/local_legends_supplement.py --source-dir revision_2026-11-09/figure2_restore_v1/figures --output-dir revision_2026-11-09/main_panels_local_legends_v1/figures

Required input files: Supplementary_Figure_S1.pdf through Supplementary_Figure_S29.pdf.
For unchanged figures, their corresponding existing SVG and PNG files are copied too.
The inputs are the authors' v30 vector figure files supplied with the manuscript
figure package. No patient records, prediction rows, clinical identifiers, or
model training inputs are read.

The script copies the original PDF drawing objects into the new layout. Curves,
confidence intervals, plotted points, bars and heatmap cells are not re-estimated.
Local vector legends and category labels are added where needed. Original figure
numbers are retained. S8 old b/d become a/b, S14 old b/c become a/b, and S15 old
a/b/d/e become a/b/c/d after the selected panels move to the main figures.

S7, S9 and S12 are delegated to local_legends_root.py and are intentionally not
written by this script. All other unmodified figures retain their source files.

Outputs:
- figures/Supplementary_Figure_SN.pdf, .svg and .png
- figures/supplement_panels/ local reviewer-response panel crops
- review/supplement_panel_bounds.json
- review/supplement_local_legend_changes.json

The PNG files are rendered previews; the publication PDF and SVG files retain
the original vector objects.

Optional check: check_supplement_plot_geometry.py requires NumPy and PyMuPDF.
It compares the copied coloured drawing coordinates after their documented
layout translations and verifies unchanged figure files by direct comparison.
