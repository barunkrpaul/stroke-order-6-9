import numpy as np, pandas as pd

def load():
    d=pd.read_csv('pd.tsv.gz',sep='\t')
    X=d.iloc[:,:16].to_numpy(np.float64); y=d['target'].to_numpy(np.int64)
    P=X.reshape(-1,8,2)
    return P[:7494],y[:7494],P[7494:],y[7494:]

def rotate(P,deg,jitter=None,rng=None):
    if jitter is not None:
        t=np.radians(rng.uniform(0,360,len(P)))
    else:
        t=np.full(len(P),np.radians(deg))
    c,s=np.cos(t),np.sin(t)
    C=P.mean(1,keepdims=True); Q=P-C
    x=Q[...,0]*c[:,None]-Q[...,1]*s[:,None]
    yy=Q[...,0]*s[:,None]+Q[...,1]*c[:,None]
    return np.stack([x,yy],-1)+C

# ---------- invariant descriptors ----------
def _unit(v):
    L=np.linalg.norm(v,axis=-1,keepdims=True)
    return v/(L+1e-9), L[...,0]

def turning(P,step=1):
    v=P[:,step:]-P[:,:-step]
    u,L=_unit(v)
    cr=u[:,:-1,0]*u[:,1:,1]-u[:,:-1,1]*u[:,1:,0]
    dt=(u[:,:-1]*u[:,1:]).sum(-1)
    return np.arctan2(cr,dt), L

def f_turn(P):
    k,L=turning(P)
    return np.concatenate([k, L/(L.sum(1,keepdims=True)+1e-9)],1)

def f_multiscale(P):
    out=[]
    for st in (1,2,3):
        k,L=turning(P,st)
        out += [k, L/(L.sum(1,keepdims=True)+1e-9)]
    return np.concatenate(out,1)

def f_dist(P):                     # all pairwise distances, scale-normalised
    D=np.linalg.norm(P[:,:,None,:]-P[:,None,:,:],axis=-1)
    iu=np.triu_indices(8,1)
    d=D[:,iu[0],iu[1]]
    return d/(d.max(1,keepdims=True)+1e-9)

def f_signed_area(P):              # signed areas of consecutive triples (chirality)
    a=P[:,:-2]; b=P[:,1:-1]; c=P[:,2:]
    ar=0.5*((b[...,0]-a[...,0])*(c[...,1]-a[...,1])-(b[...,1]-a[...,1])*(c[...,0]-a[...,0]))
    sc=np.abs(ar).max(1,keepdims=True)+1e-9
    return ar/sc

def f_full(P):
    return np.concatenate([f_multiscale(P), f_dist(P), f_signed_area(P)],1)

def f_raw(P):
    C=P.mean(1,keepdims=True); Q=P-C
    sc=np.sqrt((Q**2).sum(-1).mean(-1))[:,None,None]+1e-9
    return (Q/sc).reshape(len(P),-1)

# ablations
def f_full_nosign(P):              # reflection-invariant: signs removed
    ms=f_multiscale(P); n=ms.shape[1]
    k_idx=[]; 
    out=[]
    for st,(a,b) in zip((1,2,3),[(0,0)]*3): pass
    # rebuild with |turning| only
    parts=[]
    for st in (1,2,3):
        k,L=turning(P,st)
        parts += [np.abs(k), L/(L.sum(1,keepdims=True)+1e-9)]
    return np.concatenate(parts+[f_dist(P)],1)

def f_full_noorder(P):             # stroke direction unknown: canonicalise over reversal
    Pr=P[:,::-1,:]
    A=f_full(P); B=f_full(Pr)
    pick=(A.sum(1)>=B.sum(1))[:,None]
    return np.where(pick,A,B)

# ---- wrap-free turning representation: (sin k, cos k) instead of k ----
def f_full2(P):
    parts=[]
    for st in (1,2,3):
        k,L=turning(P,st)
        parts += [np.sin(k), np.cos(k), L/(L.sum(1,keepdims=True)+1e-9)]
    return np.concatenate(parts+[f_dist(P), f_signed_area(P)],1)

def f_full2_nosign(P):
    parts=[]
    for st in (1,2,3):
        k,L=turning(P,st)
        parts += [np.abs(np.sin(k)), np.cos(k), L/(L.sum(1,keepdims=True)+1e-9)]
    return np.concatenate(parts+[f_dist(P)],1)

def f_full2_noorder(P):
    A,B=f_full2(P),f_full2(P[:,::-1,:])
    return np.where((A.sum(1)>=B.sum(1))[:,None],A,B)
