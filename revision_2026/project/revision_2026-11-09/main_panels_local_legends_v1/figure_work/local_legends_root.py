"""Give Supplementary Figures S7, S9 and S12 panel-local legends.

Numerical marks are copied from the preceding vector PDFs without refitting.
Run with --source-dir pointing to the v30 figure PDFs and --output-dir if needed.
Requires PyMuPDF. No clinical records are read.
"""
from pathlib import Path
import argparse, json
import fitz

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--source-dir',type=Path,default=HERE.parents[1]/'figure2_restore_v1/figures')
ap.add_argument('--output-dir',type=Path,default=HERE.parent/'figures')
args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
FONT=fitz.Font('helv')
AUDIT=[]
def text(p,x,y,t,size=7.5,bold=False,center=False):
    f=fitz.Font('hebo' if bold else 'helv')
    if center:x-=f.text_length(t,fontsize=size)/2
    p.insert_text((x,y),t,fontsize=size,fontname='hebo' if bold else 'helv',color=(.1,.1,.1))
def rgb(h):return tuple(int(h[i:i+2],16)/255 for i in (1,3,5))
def legend_item(p,x,y,label,color,line=False,dash=False,size=7.5):
    c=rgb(color)
    if line:
        p.draw_line((x,y-2.6),(x+12,y-2.6),color=c,width=1.1,dashes='[3 2] 0' if dash else None)
        if not dash:p.draw_circle((x+6,y-2.6),1.8,color=c,fill=c)
    else:p.draw_rect(fitz.Rect(x,y-6.2,x+9,y-.7),color=None,fill=c,fill_opacity=.85)
    text(p,x+16,y,label,size)
def source(n):return fitz.open(args.source_dir/f'Supplementary_Figure_S{n}.pdf')
def export(d,n,panels,detail):
    p=d[0];assert not p.get_images()
    stem=f'Supplementary_Figure_S{n}'
    d.save(args.output_dir/(stem+'.pdf'),garbage=4,deflate=True)
    svg=p.get_svg_image(text_as_path=True);assert '<image' not in svg
    (args.output_dir/(stem+'.svg')).write_text(svg,encoding='utf-8')
    p.get_pixmap(matrix=fitz.Matrix(3,3),alpha=False).save(args.output_dir/(stem+'.png'))
    bounds={key:[box[0]/p.rect.width,box[1]/p.rect.height,box[2]/p.rect.width,box[3]/p.rect.height] for key,box in panels.items()}
    AUDIT.append({'figure':stem,'panel_bounds':bounds,'changes':detail,
                  'numeric_marks':'Original vector plots retained; no model fitting or numerical recomputation.',
                  'raster_images':len(p.get_images()),'size_points':list(p.rect)})

# S7: copy each original panel without the figure-wide linear/spline legend.
s=source(7);d=fitz.open();p=d.new_page(width=510,height=452)
boxes={};clips=[(0,0,252,196),(252,0,504,196),(0,217.5,252,409),(252,217.5,504,409)]
for i,c in enumerate(clips):
    col,row=i%2,i//2;x,y=col*255,row*226
    dest=fitz.Rect(x,y,x+255,y+198)
    p.show_pdf_page(dest,s,0,clip=fitz.Rect(c))
    if i:
        legend_item(p,x+59,y+214,'Linear age','#2875ab',True)
        legend_item(p,x+144,y+214,'Spline age','#c35d49',True)
    boxes['abcd'[i]]=[x,y,x+255,y+226]
export(d,7,boxes,'Independent linear-age and spline-age keys in b, c and d; removed global key.')

# S9: unchanged six plotted learning curves, each with its own five-method key.
s=source(9);d=fitz.open();p=d.new_page(width=510,height=418)
methods=[('Direct source','#454545',True),('Platt (slope unrestricted)','#976229',False),
         ('Source score + labs update','#007C83',False),('Local fit / source transform','#6B5B95',False),
         ('Fully target-local logistic','#BE3C4B',False)]
boxes={};sw=s[0].rect.width/3
for i in range(6):
    col,row=i%3,i//3;x,y=col*170,row*209
    top=0 if row==0 else 247
    clip=fitz.Rect(col*sw,top,(col+1)*sw,top+240)
    p.show_pdf_page(fitz.Rect(x,y,x+170,y+149),s,0,clip=clip)
    for k,(label,color,dash) in enumerate(methods):
        legend_item(p,x+21,y+160+k*9,label,color,True,dash,7.5)
    boxes['abcdef'[i]]=[x,y,x+170,y+209]
export(d,9,boxes,'Repeated the five-method key below each of a-f; removed the shared footer.')

# S12: remove only the old a legend, preserving its error-bar cap underneath.
s=source(12);sp=s[0]
legend_text=[]
for b in sp.get_text('dict')['blocks']:
    for line in b.get('lines',[]):
        for run in line['spans']:
            if run['text'] in ['SCH','WHU'] and 80<run['bbox'][1]<102:
                legend_text.append(run['bbox']);sp.add_redact_annot(run['bbox'],fill=None,cross_out=False)
assert len(legend_text)==2
sp.apply_redactions(images=0,graphics=0,text=0)
for box in [(175.19,86.04,187.23,90.28),(175.19,94.52,187.23,98.77)]:
    sp.add_redact_annot(box,fill=None,cross_out=False)
sp.apply_redactions(images=0,graphics=2,text=1)
d=fitz.open();p=d.new_page(width=510,height=475)
text(p,255,14,'Recorded-label proportions by feature tertile',9.5,True,True)
boxes={};sw=645.17626953125/3
for i in range(9):
    col,row=i%3,i//3;x,y=col*170,22+row*151
    top=[65,227,389][row];bottom=[225,387,556.1476][row]
    p.show_pdf_page(fitz.Rect(x,y,x+170,y+132),s,0,clip=fitz.Rect(col*sw,top,(col+1)*sw,bottom))
    if i<7:
        legend_item(p,x+50,y+143,'SCH','#2c7bb6')
        legend_item(p,x+104,y+143,'WHU','#d7191c')
    boxes['abcdefghi'[i]]=[x,y,x+170,y+151]
export(d,12,boxes,'Independent SCH/WHU keys below a-g; h and i retain their own cohort labels. Old a key removed without removing its underlying error bar.')
(HERE/'local_legends_root_audit.json').write_text(json.dumps(AUDIT,indent=2),encoding='utf-8')
print(json.dumps({'figures':[x['figure'] for x in AUDIT],'numeric_marks_preserved':True}))
