# Per-frame table for every live run: ball, laser dot, hit test (same rules as vision/live_report.py)
import json,glob,os,sys,pickle
import cv2, numpy as np
ROOT=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","vision","debug_images"))
OUT=sys.argv[1]
DOT_MIN=40
def touches(b,sh):
    x,y,r=b;H,W=sh[:2];return x-r<2 or y-r<2 or x+r>W-3 or y+r>H-3
def find_dot(g,x,y,r):
    pad=int(max(r,10)*4);x0,y0=max(0,x-pad),max(0,y-pad)
    sub=g[y0:min(g.shape[0],y+pad),x0:min(g.shape[1],x+pad)]
    if sub.size==0:return None
    _,mv,_,loc=cv2.minMaxLoc(cv2.GaussianBlur(sub,(3,3),0))
    if mv<float(np.median(sub))+DOT_MIN:return None
    return x0+loc[0],y0+loc[1],mv
tab={}
for run in sorted(glob.glob(ROOT+'/live_*')):
    rows=[json.loads(l) for l in open(run+'/log.jsonl')]
    t0=rows[0]['t']
    for r in rows:
        r['tt']=r['t']-t0; r['dot']=None; r['hit']=None
        if r['L'] and r['laser']:
            g=cv2.imread(f"{run}/{r['i']:06d}.png",0)
            if g is None: continue
            r['shape']=g.shape
            if touches(r['L'],g.shape): continue
            d=find_dot(g,*r['L'])
            if d:
                r['dot']=d; x,y,rad=r['L']
                r['hit']=bool(np.hypot(d[0]-x,d[1]-y)<=rad)
    tab[os.path.basename(run)]=rows
    has=[r for r in rows if r['hit'] is not None]
    print(os.path.basename(run),'dot frames',len(has),'hits',sum(r['hit'] for r in has),'rate %.2f'%(sum(r['hit'] for r in has)/max(1,len(has))),flush=True)
pickle.dump(tab,open(OUT,'wb'))
