"""Check copied plot drawing coordinates against the source figures."""
from pathlib import Path
import json
import fitz
import numpy as np

BASE=Path(__file__).resolve().parents[1]
SRC=BASE.parent/'figure2_restore_v1'/'figures'
OUT=BASE/'figures'
regions={
1:[([0,0,504,212],0,0),([0,212,504,592.16],0,44)],
2:[([0,0,450,658.58],0,0)],
3:[([0,0,504,399],0,0),([151.5,420,302,574],-51.5,0),([342,420,498.5,574],4,0)],
4:[([0,15,770.12,265],0,0)],
8:[([77,198,266,356],80.5,-181.3),([0,405,504,596],0,-166.8)],
14:[([321,25,613.2,216],-321,0),([321,325,613.2,518],-13.2,-302)],
15:[([0,0,504,153],0,0),([0,423,504,532],0,-233)],
16:[([0,0,504,438.13],0,0)],
17:[([0,20,470.88,597.92],0,0)],
18:[([0,51,756,684],0,0)],
27:[([0,0,504,680],0,0)],
29:[([0,0,532.8,741.6],0,0)],
}

def flatten(items,dx=0,dy=0):
    kinds=[];vals=[]
    for item in items:
        kinds.append(item[0])
        for obj in item[1:]:
            if isinstance(obj,fitz.Point):vals.extend([obj.x+dx,obj.y+dy])
            elif isinstance(obj,fitz.Rect):vals.extend([obj.x0+dx,obj.y0+dy,obj.x1+dx,obj.y1+dy])
            elif isinstance(obj,(int,float)):vals.append(float(obj))
            else:kinds.append(str(obj))
    return tuple(kinds),np.array(vals)

def coloured(path):
    c=path.get('fill') or path.get('color')
    return c is not None and max(c)-min(c)>.06

results=[]
for n,parts in regions.items():
    sd=fitz.open(SRC/f'Supplementary_Figure_S{n}.pdf')
    od=fitz.open(OUT/f'Supplementary_Figure_S{n}.pdf')
    original=sd[0].get_drawings();current=od[0].get_drawings()
    candidates=[(q,*flatten(q['items'])) for q in current if coloured(q)]
    checked=matched=0
    missing=[]
    for bounds,dx,dy in parts:
        box=fitz.Rect(bounds)
        for q in original:
            if not coloured(q) or not box.contains(q['rect']):continue
            checked+=1
            kind,values=flatten(q['items'],dx,dy)
            ok=any(kind==k and len(values)==len(v) and np.allclose(values,v,atol=.025,rtol=0) for other,k,v in candidates)
            matched+=int(ok)
            if not ok:missing.append(list(q['rect']))
    results.append({'figure':f'S{n}','source_coloured_paths_checked':checked,'same_plot_coordinates_after_documented_translation':matched,'all_checked_paths_preserved':checked==matched,'missing_rectangles':missing})
    sd.close();od.close()

unchanged=[5,6,10,11,13,19,20,21,22,23,24,25,26,28]
copies=[]
for n in unchanged:
    same=all((SRC/f'Supplementary_Figure_S{n}.{ext}').read_bytes()==(OUT/f'Supplementary_Figure_S{n}.{ext}').read_bytes() for ext in ('pdf','svg','png'))
    copies.append({'figure':f'S{n}','source_files_retained_without_changes':same})
report={'scope':'Presentation-level plot-coordinate comparison; no training or statistical re-estimation.',
        'coordinate_tolerance_points':.025,'modified_figures':results,'unchanged_figures':copies}
(BASE/'review'/'supplement_plot_geometry_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'modified':[(r['figure'],r['source_coloured_paths_checked'],r['same_plot_coordinates_after_documented_translation']) for r in results], 'unchanged_files_equal':all(r['source_files_retained_without_changes'] for r in copies)},indent=2))
