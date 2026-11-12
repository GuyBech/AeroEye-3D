import sys,time,cv2,numpy as np
import scenes2 as scenes
ts=[float(a) for a in sys.argv[2].split(',')]
cols=int(sys.argv[3]) if len(sys.argv)>3 else 3
tiles=[];t0=time.time()
for t in ts:
    tiles.append(cv2.resize(scenes.frame(int(t*30)),(640,360),interpolation=cv2.INTER_AREA))
print('s/frame',round((time.time()-t0)/len(ts),3))
while len(tiles)%cols: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[1],cv2.vconcat([cv2.hconcat(tiles[k:k+cols]) for k in range(0,len(tiles),cols)]),[cv2.IMWRITE_JPEG_QUALITY,90])
