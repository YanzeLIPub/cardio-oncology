# Figure 2 source

Run `python build_figure2.py` from this directory with PyMuPDF installed. The script writes vector SVG and PDF figures, PNG previews and panel crops to the adjacent `figures` directory.

Panels a and b display the same nine original within-centre and transfer AUCs in complementary views. Panels c–e use the 60 aggregate bin summaries supplied in `trajectory_binned_scores.csv`; no patient-level data or identifiers are included.

The curves summarize separate historical encounter-model scores, grouped by recorded label and normalized within-patient record position. They do not locate event onset or quantify advance warning. Shading is one record-level standard error. MIMIC patients may contribute admissions to both label groups, and archived MIMIC row order is not uniformly chronological. Model, grouping and sample definitions are reported in the manuscript.

The aggregate curve values reproduce the original plotted means and bands. This plotting update does not retrain the underlying models.
