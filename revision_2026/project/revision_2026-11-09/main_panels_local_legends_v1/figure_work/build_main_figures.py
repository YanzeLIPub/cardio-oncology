"""Publication Figure 2, 4 and 5 layout updates from aggregate inputs.

Run with Python and PyMuPDF installed. Inputs and outputs are relative to this
script. No models are refitted and no patient-level records are read.
"""
from pathlib import Path
import csv
import html
import json
import math
import fitz

HERE = Path(__file__).resolve().parent
INPUT = HERE / "inputs"
OUT = HERE.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
read = lambda name: list(csv.DictReader((INPUT / name).open(encoding="utf-8-sig", newline="")))
COLORS = {"SCH":"#2166AC", "WHU":"#D6604D", "MIMIC":"#4DAC26"}

class SVG:
    def __init__(self,w,h):
        self.w,self.h=w,h
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="{h/w*180}mm" viewBox="0 0 {w} {h}">',f'<rect width="{w}" height="{h}" fill="white"/>']
    def text(self,x,y,s,size=7.8,anchor="start",weight="normal",rotate=None,fill="#222222"):
        tr=f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        self.parts.append(f'<text x="{x:.6f}" y="{y:.6f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{tr}>{html.escape(str(s))}</text>')
    def rect(self,x,y,w,h,fill,stroke=None,sw=.6,opacity=1):
        self.parts.append(f'<rect x="{x:.7f}" y="{y:.7f}" width="{w:.7f}" height="{h:.7f}" fill="{fill}" fill-opacity="{opacity}"'+(f' stroke="{stroke}" stroke-width="{sw}"' if stroke else "")+'/>')
    def poly(self,points,color,fill="none",width=1.2,dash=None,opacity=1,closed=False):
        if dash:
            pattern=[float(v) for v in dash.split()];phase=0;remaining=pattern[0]
            for (x1,y1),(x2,y2) in zip(points[:-1],points[1:]):
                length=math.hypot(x2-x1,y2-y1);position=0.
                while position<length-1e-10:
                    step=min(remaining,length-position)
                    if phase%2==0:
                        self.poly([(x1+(x2-x1)*position/length,y1+(y2-y1)*position/length),(x1+(x2-x1)*(position+step)/length,y1+(y2-y1)*(position+step)/length)],color,width=width,opacity=opacity)
                    position+=step;remaining-=step
                    if remaining<1e-10: phase=(phase+1)%len(pattern);remaining=pattern[phase]
            return
        d="M "+" L ".join(f"{x:.7f},{y:.7f}" for x,y in points)+(" Z" if closed else "")
        self.parts.append(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{width}" opacity="{opacity}" stroke-linejoin="round" stroke-linecap="round"/>')
    def line(self,x1,y1,x2,y2,color="#333333",width=.65,dash=None): self.poly([(x1,y1),(x2,y2)],color,width=width,dash=dash)
    def circle(self,x,y,r,color): self.parts.append(f'<circle cx="{x:.7f}" cy="{y:.7f}" r="{r}" fill="{color}"/>')
    def save(self,name,panels):
        s="\n".join(self.parts+["</svg>"]);(OUT/(name+".svg")).write_text(s,encoding="utf-8")
        src=fitz.open(stream=s.encode(),filetype="svg");pdf=fitz.open("pdf",src.convert_to_pdf())
        save_pdf(pdf,name,panels,self.w)

def save_pdf(pdf,name,panels,width):
    pdf.save(OUT/(name+".pdf"),garbage=4,deflate=True)
    p=pdf[0];p.get_pixmap(matrix=fitz.Matrix(300/72,300/72),alpha=False).save(OUT/(name+".png"))
    p.get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False).save(OUT/(name+"_preview.png"))
    scale=p.rect.width/width
    for key,bounds in panels.items():
        crop=fitz.Rect([v*scale for v in bounds]);sub=fitz.open();q=sub.new_page(width=crop.width,height=crop.height);q.show_pdf_page(q.rect,pdf,0,clip=crop)
        sub.save(OUT/(name+"_panel_"+key+".pdf"),garbage=4,deflate=True)
        q.get_pixmap(matrix=fitz.Matrix(240/72,240/72),alpha=False).save(OUT/(name+"_panel_"+key+".png"));sub.close()
    REPORT[name]={"size_pt":list(p.rect),"raster_images":len(p.get_images()),"vector_drawings":len(p.get_drawings()),"panel_bounds":panels}

def blend(a,b,t):
    aa=[int(a[i:i+2],16) for i in (1,3,5)];bb=[int(b[i:i+2],16) for i in (1,3,5)]
    return "#"+"".join(f"{round(x+t*(y-x)):02x}" for x,y in zip(aa,bb))
def heat(v):
    t=max(0,min(1,(v-.4)/.45))
    return blend("#4575B4","#FFFFBF",t*2) if t<=.5 else blend("#FFFFBF","#D73027",t*2-1)

def figure2():
    s=SVG(510,560);centres=["SCH","WHU","MIMIC"]
    auc=[[.657,.438,.500],[.485,.840,.494],[.559,.444,.742]]
    s.text(6,16,"a",10,weight="bold");s.text(35,16,"Within- and cross-centre AUC",8.5,weight="bold")
    x,y,w,h=35,40,226,110;ym=lambda v:y+h*(.9-v)/.6
    for v in [.3,.4,.5,.6,.7,.8,.9]:
        yy=ym(v);s.line(x,yy,x+w,yy,"#E1E1E1",.45);s.line(x-3,yy,x,yy);s.text(x-6,yy+2.6,f"{v:.1f}",anchor="end")
    s.line(x,y,x,y+h);s.line(x,y+h,x+w,y+h);s.line(x,ym(.5),x+w,ym(.5),"#666666",.8,"3 2");s.text(9,y+h/2,"AUC",anchor="middle",rotate=-90)
    for target in range(3):
        mid=x+w*(target+.5)/3
        for source in range(3):
            bx=mid+(source-1)*21.5-8.25;v=auc[source][target]
            s.rect(bx,ym(v),16.5,ym(.3)-ym(v),COLORS[centres[source]],"#333333" if source==target else None,opacity=1 if source==target else .42)
            s.rect(bx-2.25,ym(v)-11,21,9,"white");s.text(bx+8.25,ym(v)-4,f"{v:.3f}",7.5,anchor="middle")
        s.text(mid,165,centres[target],anchor="middle")
    s.text(x+w/2,181,"Test centre",anchor="middle");s.text(35,197,"Training centre:")
    for i,c in enumerate(centres):s.rect(99+i*56,190.5,8,6,COLORS[c]);s.text(111+i*56,197,c)
    s.text(278,16,"b",10,weight="bold");s.text(319,16,"Cross-centre matrix",8.5,weight="bold")
    mx,my,cell=330,35,40
    for row in range(3):
        s.text(mx-8,my+(row+.5)*cell+2.6,centres[row],anchor="end")
        for col in range(3):
            v=auc[row][col];s.rect(mx+col*cell,my+row*cell,cell,cell,heat(v),"white",.7);s.text(mx+(col+.5)*cell,my+(row+.5)*cell+3.3,f"{v:.3f}",10,anchor="middle")
    for col in range(3):s.text(mx+(col+.5)*cell,171,centres[col],anchor="middle")
    s.text(mx+1.5*cell,188,"Test centre",anchor="middle");s.text(285,95,"Training centre",anchor="middle",rotate=-90)
    for j in range(180):s.rect(463,my+j/180*120,6,120/180+.25,heat(.85-(j+.5)/180*.45))
    for v in [.4,.5,.6,.7,.8]:
        yy=my+120*(.85-v)/.45;s.line(469,yy,471,yy,width=.5);s.text(474,yy+2.6,f"{v:.1f}")
    s.text(466,my-8,"AUC",anchor="middle")
    rows=read("trajectory_binned_scores.csv");ticks={"SCH":[.44,.46,.48,.50,.52],"WHU":[.30,.35,.40,.45],"MIMIC":[.35,.40,.45,.50,.55,.60]};patients={"SCH":28474,"WHU":1479,"MIMIC":1879}
    for i,c in enumerate(centres):
        px,py,pw,ph=36+i*166,273,129,102;data={lab:sorted([r for r in rows if r['center']==c and int(r['label'])==lab],key=lambda r:int(r['bin'])) for lab in [0,1]}
        lo=min(float(r['mean'])-float(r['sem']) for rr in data.values() for r in rr);hi=max(float(r['mean'])+float(r['sem']) for rr in data.values() for r in rr);span=hi-lo;lo-=.05*span;hi+=.05*span
        xx=lambda v:px+(v+.02)/1.04*pw;yy=lambda v:py+(hi-v)/(hi-lo)*ph
        s.text(px-30,216,"cde"[i],10,weight="bold");s.text(px+pw/2,216,c,8.5,anchor="middle",weight="bold");s.text(px+pw/2,229,f"{patients[c]:,} multi-record patients",7.5,anchor="middle")
        s.text(px+pw/2,240,"Recorded label",7.5,anchor="middle")
        s.line(px+10,248,px+21,248,COLORS[c],1.3);s.text(px+25,250.5,"Positive",7.5)
        s.line(px+72,248,px+83,248,COLORS[c],1.3,"3 2");s.text(px+87,250.5,"Negative",7.5)
        s.rect(px+22,257,9,5,COLORS[c],opacity=.16);s.text(px+36,262,"Shading: mean ± SEM",7.5)
        for v in ticks[c]:
            if lo<=v<=hi:s.line(px,yy(v),px+pw,yy(v),"#E1E1E1",.45);s.line(px-3,yy(v),px,yy(v));s.text(px-6,yy(v)+2.6,f"{v:.2f}",7.5,anchor="end")
        s.line(px,py,px,py+ph);s.line(px,py+ph,px+pw,py+ph)
        for v,label in [(0,"0"),(.25,"0.25"),(.5,"0.5"),(.75,"0.75"),(1,"1")]:s.line(xx(v),py+ph,xx(v),py+ph+3);s.text(xx(v),389,label,7.5,anchor="middle")
        for lab in [1,0]:
            rr=[(float(r['center_position']),float(r['mean']),float(r['sem'])) for r in data[lab]]
            band=[(xx(v),yy(m+e)) for v,m,e in rr]+[(xx(v),yy(m-e)) for v,m,e in reversed(rr)]
            s.poly(band,"none",fill=COLORS[c],width=0,opacity=.16,closed=True);s.poly([(xx(v),yy(m)) for v,m,e in rr],COLORS[c],width=1.35,dash="3 2" if lab==0 else None)
        s.text(px-29,324,"Mean model score",7.5,anchor="middle",rotate=-90);s.text(px+pw/2,403,"Normalized record position",7.5,anchor="middle");s.text(px+pw/2,415,"0 = first record; 1 = last record",7.5,anchor="middle")
    s.text(6,439,"f",10,weight="bold");s.text(35,439,"Four-laboratory transfer and local comparisons",8.5,weight="bold")
    allrows=read('figure2f_common_laboratory.csv')
    for k,group in enumerate(['transfer','local']):
        px=128+k*255;pw=119;py=468;ph=63;xx=lambda v:px+(v-.3)/.45*pw
        s.text(px+pw/2,455,'Transferred model' if k==0 else 'Local model',8,anchor='middle',weight='bold')
        for v in [.3,.4,.5,.6,.7]:
            s.line(xx(v),py-4,xx(v),py+ph,'#E1E1E1',.45);s.line(xx(v),py+ph,xx(v),py+ph+3);s.text(xx(v),542,f'{v:.1f}',7.5,anchor='middle')
        s.line(xx(.5),py-4,xx(.5),py+ph,'#777777',.7,'2 2');s.line(px,py+ph,px+pw,py+ph)
        for j,r in enumerate([r for r in allrows if r['group']==group]):
            yy=py+4+j*16;cov=r['population']=='Covered target';co='#626262' if cov else ('#4B2887' if r['target']=='MIMIC' else '#BC5639')
            prefix=r['source']+' → '+r['target'] if group=='transfer' else r['target']
            label=prefix+(': covered' if cov else ': full')
            s.text(px-7,yy+2.6,label,7.5,anchor='end');a,b,v=[float(r[n]) for n in ['ci_low','ci_high','auc']]
            s.line(xx(a),yy,xx(b),yy,co,1);s.line(xx(a),yy-2,xx(a),yy+2,co,1);s.line(xx(b),yy-2,xx(b),yy+2,co,1);s.circle(xx(v),yy,2.1,co)
        s.text(px+pw/2,555,'AUROC (95% CI)',7.5,anchor='middle')
    s.save('Figure_2',{'a':[0,0,275,204],'b':[275,0,510,204],'cde':[0,205,510,421],'f':[0,426,510,560]})

def figure5():
    s=SVG(510,560);imp=read('figure5ab_original_importance.csv')
    for k,c in enumerate(['SCH','WHU']):
        px,py,pw=112+k*255,40,132
        s.text(6+k*255,17,'ab'[k],10,weight='bold');s.text(px+pw/2,17,c+' top-25',8.5,anchor='middle',weight='bold')
        rr=sorted([r for r in imp if r['cohort']==c],key=lambda r:int(r['rank']))
        for j,r in enumerate(rr):
            yy=py+j*9;s.text(px-6,yy+2.5,r['feature'],7.5,anchor='end');s.rect(px,yy-3,132*float(r['normalised_importance']),6,'#3288BD')
        bottom=py+24*9+7;s.line(px,py-6,px,bottom);s.line(px,bottom,px+pw,bottom)
        for v in [0,.25,.5,.75,1]:s.line(px+pw*v,bottom,px+pw*v,bottom+3);s.text(px+pw*v,bottom+14,f'{v:.2f}',7.5,anchor='middle')
        s.text(px+pw/2,bottom+29,'Normalised importance',7.8,anchor='middle')
    assoc=read('figure5c_feature_associations.csv');px,py,pw,step=112,345,132,18
    s.text(6,315,'c',10,weight='bold');s.text(28,315,'Eight held-out features',8.5,weight='bold')
    for v in [-1,-.5,0,.5,1]:
        xx=px+(v+1)/2*pw;s.line(xx,py-10,xx,py+7*step+10,'#E1E1E1',.45);s.line(xx,py+7*step+10,xx,py+7*step+13);s.text(xx,py+7*step+24,f'{v:.1f}',7.5,anchor='middle')
    s.line(px+pw/2,py-10,px+pw/2,py+7*step+10,'#777777',.7,'2 2');s.line(px,py+7*step+10,px+pw,py+7*step+10)
    for j,r in enumerate(assoc):
        yy=py+j*step;s.text(px-6,yy+2.5,r['feature'],7.5,anchor='end')
        for cohort,off in [('SCH',-2.8),('WHU',2.8)]:s.circle(px+(float(r[cohort+'_rho'])+1)/2*pw,yy+off,2.4,COLORS[cohort])
    s.text(px+pw/2,510,'Observed value / SHAP rho',7.8,anchor='middle')
    s.circle(131,525,2.4,COLORS['SCH']);s.text(139,527.5,'SCH',7.8);s.circle(188,525,2.4,COLORS['WHU']);s.text(196,527.5,'WHU',7.8)
    s.text(261,315,'d',10,weight='bold');s.text(283,315,'Seed-42 feature retention',8.5,weight='bold')
    retention=read('figure5d_seed_retention.csv');dx,dy,cw,rh=368,336,33,18
    for j,r in enumerate(assoc):
        s.text(dx-6,dy+(j+.5)*rh+2.5,r['feature'],7.5,anchor='end')
        for k,seed in enumerate([42,43,44,45]):
            q=next(q for q in retention if q['feature']==r['feature'] and q['seed']==str(seed));yes=q['retained']=='Yes'
            s.rect(dx+k*cw,dy+j*rh,cw,rh,'#218C8C' if yes else '#F0F0F0','white',.35);s.text(dx+(k+.5)*cw,dy+(j+.5)*rh+2.5,q['retained'],7.8,anchor='middle',fill='white' if yes else '#222222')
    for k,seed in enumerate([42,43,44,45]):s.text(dx+(k+.5)*cw,495,str(seed),7.5,anchor='middle')
    s.text(dx+2*cw,510,'Training seed',7.8,anchor='middle')
    for yy,txt in [(526,'Yes: joint q < 0.05 in both cohorts,'),(538,'opposite signs, five-fold agreement and'),(550,'seed-42 signs retained. No: otherwise.')]:s.text(279,yy,txt,7.5)
    s.save('Figure_5',{'ab':[0,0,510,300],'c':[0,303,255,560],'d':[255,303,510,560],'cd':[0,303,510,560]})

def figure4():
    pdf=fitz.open(INPUT/'Figure_4_base.pdf');p=pdf[0]
    p.add_redact_annot(fitz.Rect(175,344.0,390,354.5),fill=(1,1,1));p.apply_redactions(images=0,graphics=2,text=0)
    p.draw_rect(p.rect,color=None,fill=(1,1,1),overlay=False)
    raw=(.7960784435,.270588249,.258823544);platt=(.288705885,.156862706,.478431404)
    for mid in [105.24,273.72,442.20]:
        left=mid-62
        p.draw_line((left,349),(left+9,349),color=raw,width=1.2);p.insert_text((left+13,351.5),'Raw base',fontname='helv',fontsize=7.5)
        p.draw_line((left+52,349),(left+61,349),color=platt,width=1.2);p.insert_text((left+65,351.5),'Held-out Platt',fontname='helv',fontsize=7.5)
    (OUT/'Figure_4.svg').write_text(p.get_svg_image(text_as_path=True),encoding='utf-8')
    save_pdf(pdf,'Figure_4',{
        'ac':[0,0,518.4,181], 'af':[0,0,518.4,356],
        'df':[0,182,518.4,356], 'g':[0,356,170.55,554.4],
        'h':[171.07,356,339.03,554.4], 'i':[339.55,356,518.4,554.4]
    },p.rect.width)

REPORT={}
if __name__=='__main__':
    figure2();figure5();figure4()
    REPORT['unchanged_main_figures']=[1,3]
    REPORT['new_analysis_performed']=False
    REPORT['promoted_panels']={'S8a + S8c':'Figure 2f','S14a':'Figure 5c','S15c':'Figure 5d'}
    REPORT['local_legend_audit']={
        'Figure 1':'Direct cohort/axis annotations and local score-distribution keys; no cross-panel key.',
        'Figure 2':'Training-centre key in a; colour scale in b; own recorded-label/SEM keys and axes in c, d and e; direct population labels in f.',
        'Figure 3':'Local keys in a, b and e; colour scales belong to c and d separately. The two heatmaps inside c share that single panel scale.',
        'Figure 4':'Each calibration panel d, e and f has its own raw-base/held-out-Platt key; other keys remain within their respective panels.',
        'Figure 5':'Direct centre/feature labels in a/b; local SCH/WHU key in c; Yes/No cell labels and a local definition in d.'}
    (HERE/'main_figure_build_report.json').write_text(json.dumps(REPORT,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in REPORT.items() if k.startswith('Figure_')},indent=2))
