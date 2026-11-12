# Phone clip: track the black ball and the red laser dot on every frame
import cv2,numpy as np,pickle,sys
# usage: python scan_phone.py <out.pkl> <phone_clip.mp4>
SRC=sys.argv[2] if len(sys.argv)>2 else 'phone_clip.mp4'
c=cv2.VideoCapture(SRC); n=int(c.get(7)); fps=c.get(5)
rows=[]
S=2  # work at half-res
for i in range(n):
    ok,f=c.read()
    if not ok: break
    small=cv2.resize(f,None,fx=1/S,fy=1/S,interpolation=cv2.INTER_AREA)
    g=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    b,gg,r=[small[:,:,k].astype(np.int16) for k in range(3)]
    # ball: darkest compact blob
    m=(g<60).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    nn,lab,st,cen=cv2.connectedComponentsWithStats(m)
    ball=None;best=0
    for k in range(1,nn):
        x,y,w,h,a=st[k]
        if a<250 or a>12000: continue
        fill=a/(w*h); ar=w/h
        if fill<.55 or ar<.6 or ar>1.6: continue
        if a>best: best=a;ball=(cen[k][0]*S,cen[k][1]*S,np.sqrt(a/np.pi)*S)
    # laser: strongest red-dominant pixel
    red=(r-(gg+b)/2).astype(np.float32); red=cv2.GaussianBlur(red,(5,5),0)
    _,mv,_,loc=cv2.minMaxLoc(red)
    dot=(loc[0]*S,loc[1]*S,float(mv)) if mv>35 else None
    rows.append(dict(i=i,t=i/fps,ball=ball,dot=dot,mean=float(g.mean())))
    if i%500==0: print(i,n,flush=True)
pickle.dump(rows,open(sys.argv[1],'wb'))
