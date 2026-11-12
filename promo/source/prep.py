# Data prep: official KPIs (same rules as vision/live_report.py), telemetry window, phone tracks, stereo pair, pipeline stages
import json,pickle,sys,os,glob
import numpy as np,cv2
from scipy.signal import savgol_filter,medfilt
VISION=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","vision"))
sys.path.insert(0,VISION)
import detector_dark as dd
ROOT=os.path.join(VISION,"debug_images")
tab=pickle.load(open('runs.pkl','rb'))
RUN='live_20261017_225129'
rows=tab[RUN]
# --- KPIs
def kpis(rows):
    seen=[e for e in rows if e['L']]
    st=np.array([e['t'] for e in seen]);sx=np.array([e['L'][:2] for e in seen],float)
    speed={}
    for k,e in enumerate(seen):
        w=np.abs(st-e['t'])<=.2
        if w.sum()>=3 and np.ptp(st[w])>0:
            speed[e['i']]=np.array([np.polyfit(st[w],sx[w,a],1)[0] for a in (0,1)])
    R=[]
    for e in rows:
        if not(e['laser'] and e['L'] and e['i'] in speed): continue
        if 'shape' not in e: continue
        x,y,r=e['L'];H,W=e['shape'][:2]
        if x-r<2 or y-r<2 or x+r>W-3 or y+r>H-3: continue
        d=(e['L'][0]-e['R'][0]) if e['R'] else None
        R.append(dict(sp=float(np.hypot(*speed[e['i']])),disp=d,dot=e['dot'],hit=e['hit']))
    def rate(sel):
        dots=[r for r in sel if r['dot'] is not None]
        return round(100*sum(bool(r['hit']) for r in dots)/max(1,len(dots)),1),len(dots)
    out=dict(all=rate(R),still=rate([r for r in R if r['sp']<15]),slow=rate([r for r in R if 15<=r['sp']<80]),fast=rate([r for r in R if r['sp']>=80]),
             near=rate([r for r in R if r['disp'] is not None and r['disp']>=22]),mid=rate([r for r in R if r['disp'] is not None and 12<=r['disp']<22]),far=rate([r for r in R if r['disp'] is not None and r['disp']<12]))
    seenany=[e for e in rows if e['L'] or e['R']]
    out['lock_seen']=round(100*sum(e['locked'] for e in seenany)/len(seenany),1)
    t=np.array([e['t'] for e in rows]);out['loop_hz']=round(2*(len(rows)-1)/(t[-1]-t[0]),1)
    return out
K={n:kpis(r) for n,r in tab.items()}
for n,v in K.items(): print(n,v)
# --- range model: Z from known ball size, then Z=K/d (system's own relation)
F=1/np.tan(np.radians(0.21))          # px, from calib3d.json dpan/dx
RB=3.5                                # cm, tennis ball + tape
pairs=np.array([(e['L'][2],e['L'][0]-e['R'][0]) for e in rows if e['L'] and e['R'] and e['locked'] and e['L'][0]-e['R'][0]>3])
Kz=float(np.median(F*RB/pairs[:,0]*pairs[:,1]))
print('F',round(F,1),'K cm*px',round(Kz,1),'Z range',np.round(np.percentile(F*RB/pairs[:,0],[5,50,95])))
# --- telemetry window (hits 99%, depth varies)
T0,T1=100.5,113.5
idx=[k for k,e in enumerate(rows) if T0<=e['tt']<=T1]
print('window frames',len(idx),'i',rows[idx[0]]['i'],rows[idx[-1]]['i'],'locked',sum(rows[k]['locked'] for k in idx),'hits',sum(bool(rows[k]['hit']) for k in idx),'dot',sum(rows[k]['hit'] is not None for k in idx))
W=[rows[k] for k in idx]
t=np.array([e['tt'] for e in W])-T0
x=np.array([e['L'][0] if e['L'] else np.nan for e in W],float);y=np.array([e['L'][1] if e['L'] else np.nan for e in W],float);r=np.array([e['L'][2] if e['L'] else np.nan for e in W],float)
d=np.array([e['L'][0]-e['R'][0] if e['L'] and e['R'] else np.nan for e in W],float)
def fill(a):
    n=np.arange(len(a));m=~np.isnan(a);return np.interp(n,n[m],a[m])
xf,yf,rf,df=[fill(a) for a in (x,y,r,d)]
ds=savgol_filter(medfilt(df,5),9,2)
Z=Kz/np.maximum(ds,3)
Zs=savgol_filter(Z,9,2)
X=(xf-160)*Zs/F;Y=(120-yf)*Zs/F
vX,vY,vZ=[savgol_filter(a,7,2,deriv=1,delta=np.median(np.diff(t))) for a in (X,Y,Zs)]
sent=np.array([e['sent'] for e in W],float)
dot=[e['dot'] for e in W];hit=[e['hit'] for e in W]
dd_=dict(t=t,x=xf,y=yf,r=rf,d=ds,Z=Zs,X=X,Y=Y,vX=vX,vY=vY,vZ=vZ,sent=sent,dot=dot,hit=hit,locked=np.array([e['locked'] for e in W]),i=np.array([e['i'] for e in W]),raw_d=df)
print('Z',Zs.min().round(1),Zs.max().round(1),'Y',Y.min().round(1),Y.max().round(1),'speed',np.percentile(np.sqrt(vX**2+vY**2+vZ**2),[50,95]).round(1),'pan',sent[:,0].min(),sent[:,0].max(),'tilt',sent[:,1].min(),sent[:,1].max())
# --- phone tracks
P=pickle.load(open('phone.pkl','rb'));n=len(P)
bx=np.array([r['ball'][0] if r['ball'] else np.nan for r in P]);by=np.array([r['ball'][1] if r['ball'] else np.nan for r in P]);br=np.array([r['ball'][2] if r['ball'] else np.nan for r in P])
dx=np.array([r['dot'][0] if r['dot'] else np.nan for r in P]);dy=np.array([r['dot'][1] if r['dot'] else np.nan for r in P]);dv=np.array([r['dot'][2] if r['dot'] else 0 for r in P])
def clean(a,ref=None):
    b=a.copy();m=~np.isnan(b);k=np.arange(n);b=np.interp(k,k[m],b[m]);b=medfilt(b,5);return savgol_filter(b,9,2)
phone=dict(bx=clean(bx),by=clean(by),br=clean(br),seen=~np.isnan(bx),dx=dx,dy=dy,dv=dv,bx_raw=bx,by_raw=by,br_raw=br)
# --- stereo pair
best=None;det=dd.DarkBallDetector()
for f in sorted(glob.glob(ROOT+'/check/*_L.png'))+sorted(glob.glob(ROOT+'/check_close/*_L.png')):
    a=cv2.imread(f,0);b=cv2.imread(f.replace('_L','_R'),0)
    ta=det.detect(a);tb=det.detect(b)
    if not ta or not tb: continue
    dsp=ta.x-tb.x
    ok=ta.x-ta.radius>12 and tb.x-tb.radius>12 and ta.x+ta.radius<308 and tb.x+tb.radius<308 and ta.y-ta.radius>10 and ta.y+ta.radius<230
    if ok:
        s=abs(ta.y-90)*.2+abs(dsp-34)
        if best is None or s<best[0]: best=(s,f,ta.x,ta.y,ta.radius,tb.x,tb.y,tb.radius)
print('stereo',best)
pickle.dump(dict(K=K,tele=dd_,phone=phone,stereo=best,Kz=Kz,F=F,RUN=RUN,T0=T0),open('data.pkl','wb'))
