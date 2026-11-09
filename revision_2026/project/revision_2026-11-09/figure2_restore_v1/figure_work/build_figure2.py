"""Draw the restored Figure 2 from published AUCs and aggregate record scores.

Run: python build_figure2.py
Dependencies: PyMuPDF (fitz). All figure inputs are aggregate; no patient rows.
The curves describe record position, not time before an event.
"""

from pathlib import Path
import csv
import html
import json
import math
import fitz

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
WIDTH, HEIGHT = 510.0, 460.0
CENTRES = ["SCH", "WHU", "MIMIC"]
COLORS = ["#2166AC", "#D6604D", "#4DAC26"]
AUC = [[0.657, 0.438, 0.500], [0.485, 0.840, 0.494], [0.559, 0.444, 0.742]]
PATIENTS = {"SCH": 28474, "WHU": 1479, "MIMIC": 1879}
ROWS = list(csv.DictReader((HERE / "trajectory_binned_scores.csv").open(encoding="utf-8-sig", newline="")))
SERIES = {}
for centre in CENTRES:
    for label in (0, 1):
        rows = sorted([r for r in ROWS if r["center"] == centre and int(r["label"]) == label], key=lambda r: int(r["bin"]))
        assert len(rows) == 10
        SERIES[(centre, label)] = [(float(r["center_position"]), float(r["mean"]), float(r["sem"])) for r in rows]

PARTS = [f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="{HEIGHT / WIDTH * 180:.6f}mm" viewBox="0 0 {WIDTH} {HEIGHT}">',
         '<rect width="510" height="460" fill="white"/>']

def text(x, y, content, size=7.5, anchor="start", weight="normal", rotate=None, fill="#222222"):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    PARTS.append(f'<text x="{x:.5f}" y="{y:.5f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"{transform}>{html.escape(str(content))}</text>')

def line(x1, y1, x2, y2, color="#333333", width=0.65, dash=None, opacity=1):
    if dash:
        poly([(x1,y1),(x2,y2)],color,width=width,dash=dash,opacity=opacity)
    else:
        PARTS.append(f'<line x1="{x1:.6f}" y1="{y1:.6f}" x2="{x2:.6f}" y2="{y2:.6f}" stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>')

def rect(x, y, w, h, fill, stroke=None, sw=0.6, opacity=1):
    PARTS.append(f'<rect x="{x:.6f}" y="{y:.6f}" width="{w:.6f}" height="{h:.6f}" fill="{fill}" fill-opacity="{opacity}"' + (f' stroke="{stroke}" stroke-width="{sw}"' if stroke else "") + '/>')

def poly(points, color, fill="none", width=1.4, dash=None, opacity=1, closed=False):
    if dash:
        # Explicit dash segments retain line style in SVG-to-PDF conversion.
        pattern = [float(value) for value in dash.split()]
        phase, remaining = 0, pattern[0]
        for (x1,y1),(x2,y2) in zip(points[:-1],points[1:]):
            length = math.hypot(x2-x1,y2-y1)
            position = 0.0
            while position < length-1e-10:
                step = min(remaining,length-position)
                if phase % 2 == 0:
                    start = (x1+(x2-x1)*position/length,y1+(y2-y1)*position/length)
                    end = (x1+(x2-x1)*(position+step)/length,y1+(y2-y1)*(position+step)/length)
                    poly([start,end],color,width=width,opacity=opacity)
                position += step
                remaining -= step
                if remaining < 1e-10:
                    phase = (phase+1)%len(pattern)
                    remaining = pattern[phase]
        return
    d = "M " + " L ".join(f"{x:.7f},{y:.7f}" for x, y in points) + (" Z" if closed else "")
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    PARTS.append(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{width}" opacity="{opacity}" stroke-linejoin="round" stroke-linecap="round"{extra}/>')

def blend(a, b, t):
    aa, bb = [int(a[i:i+2],16) for i in (1,3,5)], [int(b[i:i+2],16) for i in (1,3,5)]
    return "#" + "".join(f"{round(x + t*(y-x)):02x}" for x,y in zip(aa,bb))

def auc_color(v):
    t = max(0.0, min(1.0, (v - 0.4) / 0.45))
    return blend("#4575B4", "#FFFFBF", 2*t) if t <= 0.5 else blend("#FFFFBF", "#D73027", 2*t-1)

# A. The same historical AUCs as panel B, displayed by target centre.
text(6, 16, "a", 10, weight="bold")
text(35, 16, "Within- and cross-centre AUC", 8.5, weight="bold")
ax, ay, aw, ah = 35.0, 40.0, 226.0, 135.0
mapa = lambda y: ay + ah * (0.9-y) / 0.6
for tick in (0.3,0.4,0.5,0.6,0.7,0.8,0.9):
    py = mapa(tick)
    line(ax, py, ax+aw, py, "#E1E1E1", 0.45)
    line(ax-3, py, ax, py)
    text(ax-6, py+2.5, f"{tick:.1f}", anchor="end")
line(ax, ay, ax, ay+ah)
line(ax, ay+ah, ax+aw, ay+ah)
line(ax, mapa(0.5), ax+aw, mapa(0.5), "#666666", 0.8, "3 2")
text(9, ay+ah/2, "AUC", anchor="middle", rotate=-90)
for target in range(3):
    center = ax + aw*(target+0.5)/3
    for source in range(3):
        bx = center + (source-1)*21.5 - 8.25
        value = AUC[source][target]
        rect(bx, mapa(value), 16.5, mapa(0.3)-mapa(value), COLORS[source], "#333333" if source == target else None, 0.6, 1 if source == target else 0.42)
        rect(bx-2.25, mapa(value)-11, 21, 9, "white")
        text(bx+8.25, mapa(value)-4, f"{value:.3f}", anchor="middle")
    text(center, ay+ah+15, CENTRES[target], anchor="middle")
text(ax+aw/2, 205, "Test centre", anchor="middle")
text(35, 222, "Training centre:")
for i, c in enumerate(CENTRES):
    bx = 97+i*56
    rect(bx, 215.5, 8, 6, COLORS[i])
    text(bx+12, 222, c)

# B. Original 3 x 3 matrix; confidence intervals are reported in the legend.
text(278, 16, "b", 10, weight="bold")
text(319, 16, "Cross-centre matrix", 8.5, weight="bold")
mx, my, cell = 330.0, 40.0, 45.0
for row in range(3):
    text(mx-8, my+(row+0.5)*cell+2.5, CENTRES[row], anchor="end")
    for col in range(3):
        v = AUC[row][col]
        rect(mx+col*cell, my+row*cell, cell, cell, auc_color(v), "white", 0.7)
        text(mx+(col+0.5)*cell, my+(row+0.5)*cell+3.3, f"{v:.3f}", 10, anchor="middle")
for col in range(3):
    text(mx+(col+0.5)*cell, my+3*cell+15, CENTRES[col], anchor="middle")
text(mx+1.5*cell, 205, "Test centre", anchor="middle")
text(285, my+1.5*cell, "Training centre", anchor="middle", rotate=-90)
for j in range(180):
    v = 0.85 - (j+0.5)/180*0.45
    rect(475, my+j/180*135, 6, 135/180+0.25, auc_color(v))
for tick in (0.4,0.5,0.6,0.7,0.8):
    py = my+135*(0.85-tick)/0.45
    line(481, py, 483, py, width=0.5)
    text(486, py+2.5, f"{tick:.1f}")
text(478, my-8, "AUC", anchor="middle")

# Shared legend for C-E. Positive/negative refer to recorded labels.
text(106, 246, "Recorded label:")
line(167, 243.5, 185, 243.5, "#333333", 1.4)
text(190, 246, "Positive")
line(234, 243.5, 252, 243.5, "#333333", 1.4, "3 2")
text(257, 246, "Negative")
rect(306, 239.5, 13, 6, "#777777", opacity=0.2)
text(324, 246, "Mean ± SEM")

# C-E use the original Matplotlib autoscale limits (5% margins) and x limits.
ylim_audit = {}
ticks = {"SCH": [0.44,0.46,0.48,0.50,0.52], "WHU": [0.30,0.35,0.40,0.45], "MIMIC": [0.35,0.40,0.45,0.50,0.55,0.60]}
for i, centre in enumerate(CENTRES):
    px, py, pw, ph = 36.0+i*166.0, 288.0, 129.0, 116.0
    lower = min(mean-sem for label in (0,1) for _,mean,sem in SERIES[(centre,label)])
    upper = max(mean+sem for label in (0,1) for _,mean,sem in SERIES[(centre,label)])
    span = upper-lower
    ymin, ymax = lower-0.05*span, upper+0.05*span
    ylim_audit[centre] = [ymin,ymax]
    mapx = lambda x: px + (x+0.02)/1.04*pw
    mapy = lambda y: py + (ymax-y)/(ymax-ymin)*ph
    text(px-30, 265, "cde"[i], 10, weight="bold")
    text(px+pw/2, 265, centre, 8.5, anchor="middle", weight="bold")
    text(px+pw/2, 278, f'{PATIENTS[centre]:,} multi-record patients', anchor="middle")
    for tick in ticks[centre]:
        if ymin <= tick <= ymax:
            yy = mapy(tick)
            line(px, yy, px+pw, yy, "#E1E1E1", 0.45)
            line(px-3, yy, px, yy)
            text(px-6, yy+2.5, f"{tick:.2f}", anchor="end")
    line(px, py, px, py+ph)
    line(px, py+ph, px+pw, py+ph)
    for tick, label in ((0,"0"),(.25,"0.25"),(.5,"0.5"),(.75,"0.75"),(1,"1")):
        xx = mapx(tick)
        line(xx, py+ph, xx, py+ph+3)
        text(xx, py+ph+14, label, anchor="middle")
    for label in (1,0):
        data = SERIES[(centre,label)]
        band = [(mapx(x),mapy(m+s)) for x,m,s in data] + [(mapx(x),mapy(m-s)) for x,m,s in reversed(data)]
        poly(band, "none", fill=COLORS[i], width=0, opacity=0.16, closed=True)
        poly([(mapx(x),mapy(m)) for x,m,s in data], COLORS[i], width=1.4, dash="3 2" if label == 0 else None)
    text(px+pw/2, 433, "Normalized record position", anchor="middle")
text(8, 346, "Mean model score", anchor="middle", rotate=-90)
text(255, 452, "0 = first record; 1 = last record", anchor="middle")
PARTS.append("</svg>")
svg = "\n".join(PARTS)
svg_path = OUT / "Figure_2.svg"
svg_path.write_text(svg, encoding="utf-8")
doc = fitz.open(stream=svg.encode(), filetype="svg")
pdf = fitz.open("pdf", doc.convert_to_pdf())
pdf.set_metadata({"title":"Figure 2: cross-centre performance and record-position score profiles", "subject":"Restored panels A-E using the original aggregate data", "author":""})
pdf.save(OUT / "Figure_2.pdf", garbage=4, deflate=True)
page = pdf[0]
page.get_pixmap(matrix=fitz.Matrix(300/72,300/72), alpha=False).save(OUT / "Figure_2.png")
page.get_pixmap(matrix=fitz.Matrix(1.75,1.75), alpha=False).save(OUT / "Figure_2_preview.png")
panel_rects = {"a":[0,0,275,232],"b":[275,0,510,232],"c":[0,250,174,460],"d":[174,250,341,460],"e":[341,250,510,460],"cde":[0,234,510,460]}
scale = page.rect.width/WIDTH
for name, bounds in panel_rects.items():
    clip = fitz.Rect([v*scale for v in bounds])
    panel = fitz.open()
    p = panel.new_page(width=clip.width,height=clip.height)
    p.show_pdf_page(p.rect,pdf,0,clip=clip)
    panel.save(OUT / f"Figure_2_panel_{name}.pdf",garbage=4,deflate=True)
    p.get_pixmap(matrix=fitz.Matrix(200/72,200/72),alpha=False).save(OUT / f"Figure_2_panel_{name}.png")
    panel.close()
(HERE / "figure2_build_summary.json").write_text(json.dumps({
    "auc_matrix_source_by_target": AUC,
    "aggregate_curve_rows":len(ROWS),
    "curves":6,
    "bins_per_curve":10,
    "interval":"mean plus/minus one unadjusted record-level SEM",
    "normalized_record_position_limits":[-0.02,1.02],
    "original_autoscale_y_limits":ylim_audit,
    "figure_size_mm":[180,HEIGHT/WIDTH*180],
    "panel_rects_in_viewbox_points":panel_rects,
    "pdf_raster_images":len(page.get_images()),
    "pdf_vector_drawings":len(page.get_drawings()),
    "minimum_label_size_in_viewbox_pt":7.5,
    "analysis_values_changed":False
},indent=2),encoding="utf-8")
print(json.dumps({"outputs":[str(OUT / x) for x in ("Figure_2.svg","Figure_2.pdf","Figure_2.png")],"pdf_raster_images":len(page.get_images()),"curve_rows":len(ROWS)},indent=2))
