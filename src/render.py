import numpy as np
from skimage.draw import line_aa
from scipy.ndimage import gaussian_filter

SZ=32; R=12.0
def to_canvas(P):
    """centre at canvas centre, isotropic scale (max radius = R). No rotation-dependent
    normalisation, so rotating the trajectory rotates the image exactly."""
    C=P.mean(1,keepdims=True); Q=P-C
    rad=np.linalg.norm(Q,axis=-1).max(1)[:,None,None]+1e-9
    return Q*(R/rad)+SZ/2.0

def render(P):
    Q=to_canvas(P); out=np.zeros((len(P),SZ,SZ),np.float32)
    for i,q in enumerate(Q):
        img=np.zeros((SZ,SZ),np.float32)
        pix=np.clip(np.round(q).astype(int),0,SZ-1)
        for a,b in zip(pix[:-1],pix[1:]):
            rr,cc,val=line_aa(a[1],a[0],b[1],b[0])
            img[rr,cc]=np.maximum(img[rr,cc],val)
        out[i]=gaussian_filter(img,0.6)
    m=out.reshape(len(P),-1).max(1)[:,None,None]+1e-9
    return out/m
