"""Recompose existing vector supplement panels and make their keys local.

Only presentation changes: the source PDF drawing objects are copied directly.
No patient data, model fitting, coordinate inference, or numerical re-estimation.
"""
from pathlib import Path
import argparse
import json
import shutil
import fitz

BASE = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--source-dir', type=Path, default=BASE.parent/'figure2_restore_v1'/'figures')
ap.add_argument('--output-dir', type=Path, default=BASE/'figures')
args = ap.parse_args()
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)
REVIEW = BASE/'review'
REVIEW.mkdir(exist_ok=True)
ASSETS = OUT/'supplement_panels'
ASSETS.mkdir(exist_ok=True)
maps = {}
records = []

def rgb(value):
    return tuple(int(value[i:i+2],16)/255 for i in (1,3,5))

def source(n):
    return fitz.open(args.source_dir/f'Supplementary_Figure_S{n}.pdf')

def page(n, height=None, width=None):
    src=source(n); out=fitz.open()
    p=out.new_page(width=width or src[0].rect.width,height=height or src[0].rect.height)
    return src,out,p

def copy(p,src,clip=None,dest=None):
    clip=fitz.Rect(clip or src[0].rect)
    dest=fitz.Rect(dest or clip)
    p.show_pdf_page(dest,src,0,clip=clip,keep_proportion=False)

def white(p,box):
    p.draw_rect(fitz.Rect(box),color=None,fill=(1,1,1),overlay=True)

def legend(p,x,y,items,size=7,gap=10,kind='line'):
    for i,item in enumerate(items):
        text,color,*style=item
        yy=y+i*gap
        col=rgb(color)
        if kind=='box':
            p.draw_rect(fitz.Rect(x,yy-4,x+9,yy+1),color=None,fill=col)
        else:
            p.draw_line((x,yy-1.6),(x+11,yy-1.6),color=col,width=1.2,dashes='[3 2]' if style and style[0]=='dashed' else None)
            p.draw_circle((x+5.5,yy-1.6),1.5,color=None,fill=col)
        p.insert_text((x+15,yy+1),text,fontsize=size,color=(0,0,0))

def rename(p,src,old,new,clip,dest):
    c=fitz.Rect(clip); d=fitz.Rect(dest)
    matches=[]
    for block in src[0].get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for s in line['spans']:
                if s['text'].startswith(old+')') and fitz.Rect(s['bbox']).intersects(c):
                    matches.append(s)
    if not matches:
        raise RuntimeError(f'Panel letter {old} missing')
    s=matches[0]; sx=d.width/c.width; sy=d.height/c.height
    x=d.x0+(s['bbox'][0]-c.x0)*sx; y=d.y0+(s['bbox'][1]-c.y0)*sy
    h=(s['bbox'][3]-s['bbox'][1])*sy
    w=s['size']*sx*1.13
    white(p,(x-1,y-1,x+w,y+h+1))
    baseline=d.y0+(s['origin'][1]-c.y0)*sy
    p.insert_text((x,baseline),new+')',fontname='hebo',fontsize=s['size']*sy)

def save(n,src,out,p,change,panels=None,semantics=None):
    stem=f'Supplementary_Figure_S{n}'
    pdf=OUT/(stem+'.pdf')
    out.save(pdf,garbage=4,deflate=True)
    (OUT/(stem+'.svg')).write_text(p.get_svg_image(text_as_path=True),encoding='utf-8')
    p.get_pixmap(matrix=fitz.Matrix(3,3),alpha=False).save(OUT/(stem+'.png'))
    mapping=panels or {}
    maps[str(n)]={'page_size':[p.rect.width,p.rect.height],'panels':mapping}
    for key,bounds in mapping.items():
        p.get_pixmap(matrix=fitz.Matrix(3,3),clip=fitz.Rect(bounds),alpha=False).save(ASSETS/f'S{n}{key}.png')
        crop=fitz.open(); cp=crop.new_page(width=bounds[2]-bounds[0],height=bounds[3]-bounds[1])
        cp.show_pdf_page(cp.rect,out,0,clip=fitz.Rect(bounds))
        crop.save(ASSETS/f'S{n}{key}.pdf',garbage=4,deflate=True)
        (ASSETS/f'S{n}{key}.svg').write_text(cp.get_svg_image(text_as_path=True),encoding='utf-8')
        crop.close()
    records.append({'figure':f'S{n}','change':change,'colour_semantics':semantics,'method':'Original PDF vector drawing objects copied; local keys, category labels and panel positions edited only.','panels':mapping})
    src.close();out.close()

# S1: retain category labels in a; supply the same model keys in b/c.
s,o,p=page(1,height=636.16)
copy(p,s,[0,0,504,212],[0,0,504,212]);copy(p,s,[0,212,504,592.16],[0,256,504,636.16])
models=[('XGBoost + RF','#007C91'),('Logistic','#4E3178'),('Age only','#777777'),('Missingness only','#B44748'),('XGBoost','#4F78B5'),('Random forest','#A6761D')]
for x in (211,359):
    legend(p,x,214,models[:3],size=5.9,gap=11)
    legend(p,x+65,214,models[3:],size=5.9,gap=11)
save(1,s,o,p,'Model labels made local to b/c; all panels retained.',{'a':[0,26,205,211],'b':[205,26,354,250],'c':[354,26,504,250],'d':[0,270,258,430],'e':[258,270,504,430],'f':[0,448,504,636.16]},dict(models))

# S2: duplicate the source colour ramp, retaining its exact endpoints.
s,o,p=page(2);copy(p,s)
white(p,[450,13,503,207])
for top,bottom in ((27.2,105.7),(150.8,190.0)):
    copy(p,s,[453.3,27.3,461.2,189.9],[453.3,top,461.2,bottom])
    p.draw_rect(fitz.Rect(453.3,top,461.2,bottom),color=(0,0,0),width=.4)
    p.insert_text((449.5,top-5),'Missing (%)',fontsize=5.8)
    for value in (0,50,100):
        yy=bottom-(bottom-top)*value/100
        p.draw_line((461.2,yy),(463.1,yy),color=(0,0,0),width=.4)
        p.insert_text((465,yy+2),str(value),fontsize=5.8)
save(2,s,o,p,'Separate 0–100% missingness colourbar for each heatmap.',{'a':[0,11,503,125],'b':[0,133,503,211],'c':[0,250,255,447],'d':[255,250,504,447],'e':[0,450,255,658.58],'f':[255,450,504,658.58]},'Each a/b ramp: missing before imputation (%) from 0 to 100.')

# S3: use two-line category labels to make both outcome-group panels self-contained.
s,o,p=page(3);copy(p,s,[0,0,504,399],[0,0,504,399])
cats=['Original union\n(n=4,462)','MI (n=1,131)','Stroke (n=1,069)','Heart failure\n(n=2,847)','Cardiac arrest\n(n=418)','Cardiomyopathy\n(n=371)','Unstable angina\n(n=245)','Union without CM-specific\n(n=4,374)']
for letter,x0,clip in [('e',0,[151.5,399,302,606.09]),('f',252,[342,399,498.5,606.09])]:
    left=100 if letter=='e' else 94
    copy(p,s,clip,[x0+left,399,x0+250.5,606.09])
    for i,label in enumerate(cats):
        yy=431.4+i*18.38
        box=fitz.Rect(x0+1,yy-8,x0+96,yy+9)
        p.insert_textbox(box,label,fontsize=6.3,align=2,lineheight=1.03)
save(3,s,o,p,'Code-group names and counts shown separately in e/f; their axes and estimates retained.',{'abcd':[0,0,504,397],'e':[0,399,252,606.09],'f':[252,399,504,606.09]},'Forest points remain centre or code-group estimates as directly labelled.')

# S4: move the original comparison key into the a panel area.
s,o,p=page(4);copy(p,s);white(p,[0,281,770.12,315.62])
legend(p,65,285,[('Matched nested fixed','#454545'),('Nested selected','#976229'),('Previous validation','#007C83')],size=8,gap=11)
save(4,s,o,p,'a comparison key confined below a; b remains a directly named paired contrast.',{'a':[0,0,390,315.62],'b':[390,0,770.12,315.62]},'a: grey matched fixed, brown nested selected, teal previous validation. b: teal paired selected-minus-fixed contrast.')

# S8: old a/c promoted to main Figure 2f; keep only old b/d.
s,o,p=page(8,height=466)
bclip=[77,181.3,266,382];bdest=[157.5,0,346.5,200.7]
dclip=[0,388.8,504,631.33];ddest=[0,222,504,464.53]
copy(p,s,bclip,bdest);copy(p,s,dclip,ddest)
rename(p,s,'b','a',bclip,bdest);rename(p,s,'d','b',dclip,ddest)
white(p,[150,0,504,14])
p.insert_text((157.5,10.5),'a)  Four accepted laboratory inputs',fontname='hebo',fontsize=9.2)
save(8,s,o,p,'old b→a; old d→b. Old a/c moved to main Figure 2f.',{'a':[150,0,405,211],'b':[0,220,504,466]},'a: WHU orange, MIMIC purple; b: full target blue, covered target grey.')

# S14: old a promoted to main Figure 5c; old b/c become a/b.
s,o,p=page(14,height=310,width=600)
for old,new,clip,dest in [('b','a',[321,0,613.2,294],[0,0,292.2,294]),('c','b',[321,302,613.2,555.6],[307.8,0,600,253.6])]:
    copy(p,s,clip,dest);rename(p,s,old,new,clip,dest)
save(14,s,o,p,'old b→a; old c→b. Old a moved to main Figure 5c.',{'a':[0,0,300,310],'b':[300,0,600,310]},'a: positive teal, negative red, not-testable grey. b: centre indicated directly in each category.')

# S15: old c promoted to main Figure 5d; retain a/b and reletter d/e.
s,o,p=page(15,height=360)
copy(p,s,[0,0,504,153],[0,0,504,153])
clip=[0,400,504,589.99];dest=[0,167,504,356.99]
copy(p,s,clip,dest);rename(p,s,'d','c',clip,dest);rename(p,s,'e','d',clip,dest)
save(15,s,o,p,'a/b retained; old d→c and old e→d. Old c moved to main Figure 5d.',{'a':[125,0,285,153],'b':[318,0,482,153],'c':[0,164,259,360],'d':[259,164,504,360]},'a/b: numeric correlations printed in every cell; c: SCH blue and WHU red; d: centre colours and feature-specific solid/dashed lines locally keyed.')

# S16: local calibration keys; duplicate category names beside slope panel.
s,o,p=page(16);copy(p,s);white(p,[0,21,504,33])
cal=[('Full-training raw','#2166AC'),('Calibration-base raw','#B35806','dashed'),('Calibration-base Platt','#411B78')]
for x in (43,207,371):
    legend(p,x,57,cal,size=5.7,gap=9)
for block in s[0].get_text('dict')['blocks']:
    for line in block.get('lines',[]):
        txt=''.join(z['text'] for z in line['spans'])
        if ' / ' in txt and line['bbox'][0]<75 and line['bbox'][1]>230:
            yy=line['spans'][0]['origin'][1]
            p.insert_textbox(fitz.Rect(172,yy-7,228,yy+3),txt,fontsize=5.8,align=2)
save(16,s,o,p,'a/b/c each show all three calibration-method keys; slope e repeats its centre/model category labels.',{'a':[0,35,172,176],'b':[172,35,336,176],'c':[336,35,504,176],'de':[0,214,337,417],'d':[0,214,171,417],'e':[171,214,337,417],'f':[337,214,504,438.13]},'Calibration: blue full-training raw, orange base raw, purple base Platt. f retains its local NRI key.')

# S17: all simulation comparisons receive independent keys.
s,o,p=page(17);copy(p,s);white(p,[0,0,470.88,20])
sim=[('Ensemble','#0072B2'),('Age only','#D55E00'),('Random review','#777777','dashed')]
for x,y in ((56,50),(289,50),(56,239),(287,268)):
    legend(p,x,y,sim,size=6.5,gap=10)
save(17,s,o,p,'Independent ensemble/age/random keys in a–d; existing e/f keys retained.',{'a':[0,21,237,210],'b':[237,21,470.88,210],'c':[0,211,237,398],'d':[237,211,470.88,398],'e':[0,399,237,597.92],'f':[237,399,470.88,597.92]},'a–d: ensemble blue, age-only orange, random-review grey; e/f retain their distinct scenario/comparator keys.')

# S18: both MIMIC panels identify both input sets locally.
s,o,p=page(18);copy(p,s);white(p,[0,33,756,51])
mm=[('Original 43 inputs','#2166AC'),('Exclude AST / HDL','#1B7837')]
for x in (79,442):legend(p,x,84,mm,size=8,gap=13)
for b in s[0].get_text('dict')['blocks']:
    for line in b.get('lines',[]):
        txt=''.join(z['text'] for z in line['spans'])
        if txt.startswith('0.') and line['bbox'][0]<65 and 70<line['bbox'][1]<265:
            y=line['spans'][0]['origin'][1]
            p.insert_text((425-fitz.get_text_length(txt,fontsize=8),y),txt,fontsize=8)
save(18,s,o,p,'Local input-set keys in a/b and independent y tick labels in b; c/d unchanged.',{'a':[0,52,389,294],'b':[389,52,756,294],'cd':[0,359,756,684],'c':[0,359,389,684],'d':[389,359,756,684]},'a/b blue = original 43 inputs, green = AST/HDL excluded; c retains logistic and ensemble key; d directly labels counts.')

# S27: retain the aligned category axis; local colour keys avoid a shared legend.
s,o,p=page(27);copy(p,s)
for x in (253,392):
    legend(p,x,452,[('SCH','#2875AB')],size=5.7)
    legend(p,x+44,452,[('WHU','#C35D49')],size=5.7)
save(27,s,o,p,'d/e each receive a local centre-colour key. Original aligned category axis and compact three-column forest layout retained.',{'a':[0,0,504,295],'b':[0,295,504,440],'c':[0,440,221,680],'d':[245,440,358,680],'e':[383,440,504,680],'bcde':[0,295,504,680]},'d/e: SCH blue and WHU red; full cancer-group categories retain their common row alignment with c. a/b retain existing local legends.')

# S29: put the original four-entry decision-curve legend in every left-hand panel.
s,o,p=page(29);copy(p,s)
# Exact original legend geometry is reused rather than re-estimated or recoloured.
legend_clip=[49,108,137,148]
for dy in (185.0,370.0,555.0):
    copy(p,s,legend_clip,[49,108+dy,137,148+dy])
save(29,s,o,p,'Existing a legend repeated locally in c/e/g; right panels are directly named Platt-minus-raw differences.',{'a':[0,0,270,183],'b':[270,0,532.8,183],'c':[0,185,270,368],'d':[270,185,532.8,368],'e':[0,370,270,553],'f':[270,370,532.8,553],'g':[0,555,270,741.6],'h':[270,555,532.8,741.6]},'Left: base raw blue, independent Platt orange, treat-all grey dashed, treat-none black dotted. Right: purple paired difference, shaded interval.')

edited={int(r['figure'][1:]) for r in records}
delegated={7,9,12}
for n in range(1,30):
    if n in edited|delegated:continue
    for ext in ('pdf','svg','png'):
        name=f'Supplementary_Figure_S{n}.{ext}'
        if not (OUT/name).exists():shutil.copy2(args.source_dir/name,OUT/name)
    records.append({'figure':f'S{n}','change':'No change: panels already use local legends or direct labels.','method':'Source PDF/SVG/PNG retained.'})
(REVIEW/'supplement_panel_bounds.json').write_text(json.dumps(maps,indent=2),encoding='utf-8')
(REVIEW/'supplement_local_legend_changes.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('Written',len(edited),'edited supplemental figures; S7/S9/S12 are managed by the parent agent.')
