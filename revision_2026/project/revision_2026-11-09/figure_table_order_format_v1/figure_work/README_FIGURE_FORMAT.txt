Figure lettering and line formatting

standardize_panel_letters.py reads the preceding revision's vector figure PDFs.
It standardizes visible lettering to Arial, uses bold lower-case a), b), etc.
for panel letters, and adjusts thin stroke-only paths in their original PDF
paint order. Filled data marks retain their coordinates and geometry.
pdf_line_weights.py preserves PDF clipping and the order of all drawing
operations while changing the specified stroke widths.

Panel letters are 9 pt at the planned manuscript display widths. The 9 pt
choice is a consistent presentation style, not a journal-mandated minimum.
The source figures, captions, model results and numerical values are inputs;
the scripts do not estimate statistical results or read patient records.

Required inputs:
  - Preceding revision figures/*.pdf, with its figure crop descriptions.
  - review/figure_layout_settings.json, containing final supplementary widths.
  - An installed Arial font family (regular, bold, italic and bold italic).
  - Python with PyMuPDF and pypdf.

Example (from this revision directory):
  python figure_work/standardize_panel_letters.py \
    --source-dir ../main_panels_local_legends_v1/figures \
    --output-dir figures \
    --font-file /path/to/fonts/arialbd.ttf \
    --layout-settings review/figure_layout_settings.json

Outputs are vector PDF and SVG figures plus PNGs sized for at least 300 dpi
at their planned display widths. Existing panel crops are regenerated from
the new vector figures. review/panel_letter_format_changes.json records the
presentation changes; acceptance requires the separate text and visual checks.

The three Figure 1 headings use sentence case. The MIMIC centre colour in
Figure 2 is purple, consistent with Figure 1; the matrix colour scale is
unchanged. The filled diamond in Figure S23 remains a vector symbol because
the selected Arial font does not contain that character.
