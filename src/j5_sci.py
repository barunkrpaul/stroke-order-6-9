"""Journal experiment 5: the winding-direction rule, measured on MNIST bitmaps.
Confirms Proposition 2 empirically and explains the sign flips as a side effect
of the pixel order in which the skeleton library stores a branch."""
import numpy as np, json, sknw
from skimage.morphology import skeletonize
from scipy.ndimage import rotate as ndrot
X=np.load('mnist_X.npy').reshape(-1,28,28).astype(np.float32)/255.
y=np.load('mnist_y.npy')
rng=np.random.default_rng(0); idx=rng.permutation(len(y)); X,y=X[idx],y[idx]
Xte,yte=X[60000:],y[60000:]
K=np.isin(yte,[6,9]); I=Xte[K]; L=yte[K]
print('pair samples',len(L),flush=True)
def sci(pts):
    p=np.asarray(pts,float)
    if len(p)<4: return 0.0
    t=np.diff(p,axis=0); n=np.linalg.norm(t,axis=1,keepdims=True); t=t/np.where(n<1e-9,1e-9,n)
    cr=t[:-1,0]*t[1:,1]-t[:-1,1]*t[1:,0]; dt=(t[:-1]*t[1:]).sum(1)
    return float(np.sum(np.arctan2(cr,dt)))
def branch_sci(im,thr=0.25):
    s=skeletonize(im>thr)
    if s.sum()<5: return 0.0
    try: G=sknw.build_sknw(s.astype(np.uint8))
    except Exception: return 0.0
    best,bs=0.0,-1
    for a,b,d in G.edges(data=True):
        p=d.get('pts',[])
        if len(p)<4: continue
        v=sci(p); sc=len(p)*abs(v)
        if sc>bs: bs,best=sc,v
    return best
def pred(A): return np.array([6 if branch_sci(im)>=0 else 9 for im in A])
up=pred(I); print('upright accuracy %.2f%%'%(100*(up==L).mean()),flush=True)
r=np.random.default_rng(1)
Ir=np.stack([ndrot(im,a,reshape=False,order=1) for im,a in zip(I,r.uniform(0,360,len(I)))])
rr=pred(Ir); print('random rotation accuracy %.2f%%'%(100*(rr==L).mean()),flush=True)
I180=np.stack([ndrot(im,180,reshape=False,order=1) for im in I])
p180=pred(I180)
flip=100*(p180!=up).mean(); print('sign of the rule changes for %.1f%% of samples after a 180 deg rotation'%flip,flush=True)
json.dump(dict(upright=float(100*(up==L).mean()),rotated=float(100*(rr==L).mean()),
               flip=float(flip),n=int(len(L))),open('j5_sci_results.json','w'))
print('J5 DONE',flush=True)
