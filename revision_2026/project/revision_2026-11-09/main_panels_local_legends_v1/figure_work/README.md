# Figure panel integration and local legends

This update preserves the numerical results and moves selected supplementary
panels into Figures 2 and 5. It removes their supplementary copies and makes
the figure keys local to each panel. It does not train models.

- `build_main_figures.py`: Figures 2, 4 and 5; see `README_MAIN.txt` for input provenance.
- `local_legends_supplement.py`: supplementary layout changes and panel removal; see `README_LOCAL_LEGENDS.txt`.
- `local_legends_root.py`: local legends for S7, S9 and S12. Run with
  `--source-dir` pointing to the preceding v30 vector figure directory and
  `--output-dir` pointing to the destination figure directory.
- `check_supplement_plot_geometry.py`: comparison of retained vector plot objects.

The five CSV files contain only aggregate plotted values. The layout scripts
also require the authors' preceding vector PDFs, as described in the two input
notes; those PDFs are figure assets, not patient records. Python 3 and PyMuPDF
are used for rendering; the optional geometry comparison also uses NumPy.
