"""Image-side baselines on the SAME pendigits trajectories, rendered to 32x32.
Static models (invariant or not) vs our order-aware invariant trajectory model."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, json, time, sys
from torchvision.ops import deform_conv2d
import feats as F, render

torch.set_num_threads(2)
Ptr,ytr,Pte,yte=F.load()
ANG=list(range(0,360,10))
Ite_ang={a:render.render(F.rotate(Pte,a)) for a in ANG}
Itr0=render.render(Ptr)
print('rendered', flush=True)

def loader(I,y,bs=128,shuffle=True,seed=0):
    idx=np.arange(len(y))
    if shuffle: np.random.default_rng(seed).shuffle(idx)
    for i in range(0,len(idx),bs):
        j=idx[i:i+bs]
        yield torch.tensor(I[j]).unsqueeze(1), torch.tensor(y[j])

# ---------------- models ----------------
def head(c): return nn.Sequential(nn.AdaptiveMaxPool2d(1),nn.Flatten(),nn.Linear(c,10))

class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.f=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                          nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                          nn.Conv2d(64,128,3,padding=1),nn.ReLU())
        s.h=head(128)
    def forward(s,x): return s.h(s.f(x))

class TIPool(nn.Module):
    """TI-pooling (Laptev et al. 2016): shared CNN over k rotated copies of the input,
    max over the final FEATURE VECTOR (not over unaligned spatial maps)."""
    def __init__(s,k=8):
        super().__init__(); s.k=k
        s.f=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                          nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                          nn.Conv2d(64,128,3,padding=1),nn.ReLU(),
                          nn.AdaptiveMaxPool2d(1),nn.Flatten())
        s.c=nn.Linear(128,10)
    def rot(s,x,deg):
        t=torch.tensor(np.radians(deg),dtype=torch.float32)
        th=torch.tensor([[torch.cos(t),-torch.sin(t),0.],[torch.sin(t),torch.cos(t),0.]])
        th=th.unsqueeze(0).expand(x.size(0),-1,-1)
        g=Fn.affine_grid(th,x.size(),align_corners=False)
        return Fn.grid_sample(x,g,align_corners=False)
    def forward(s,x):
        v=torch.stack([s.f(s.rot(x,360*i/s.k)) for i in range(s.k)],1)
        return s.c(v.max(1).values)

class RICC(nn.Module):
    """Rotation-Invariant Coordinate Convolution (Mo & Zhao 2024), Eq. (6)-(7),
    implemented with deform_conv2d and offsets precomputed per spatial position."""
    def __init__(s,cin,cout):
        super().__init__()
        s.w=nn.Parameter(torch.empty(cout,cin,3,3)); nn.init.kaiming_uniform_(s.w,a=5**.5)
        s.b=nn.Parameter(torch.zeros(cout)); s.cache={}
    def offsets(s,H,W,dev):
        if (H,W) in s.cache: return s.cache[(H,W)]
        p=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
        cy,cx=(H-1)/2.,(W-1)/2.
        yy,xx=np.meshgrid(np.arange(H)-cy,np.arange(W)-cx,indexing='ij')
        phi=np.arctan2(-yy,xx)                      # direction of X0 from image centre
        off=np.zeros((1,18,H,W),np.float32)
        for k,(dy,dx) in enumerate(p):
            if dy==0 and dx==0: continue
            th=np.arctan2(-dy,dx)                   # angle of the regular-grid neighbour
            qy=-np.sin(phi+th); qx=np.cos(phi+th)   # rotated sample, radius 1
            off[0,2*k]  =qy-dy
            off[0,2*k+1]=qx-dx
        t=torch.tensor(off,device=dev); s.cache[(H,W)]=t; return t
    def forward(s,x):
        o=s.offsets(x.shape[2],x.shape[3],x.device).expand(x.size(0),-1,-1,-1)
        return deform_conv2d(x,o,s.w,s.b,padding=(1,1))

class RICCNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.l1,s.l2,s.l3=RICC(1,32),RICC(32,64),RICC(64,128)
        s.h=head(128)
    def forward(s,x):
        x=Fn.relu(s.l1(x)); x=Fn.max_pool2d(x,2)
        x=Fn.relu(s.l2(x)); x=Fn.max_pool2d(x,2)
        x=Fn.relu(s.l3(x))
        return s.h(x)

# ---------------- train / eval ----------------
def train(make,aug,epochs,seed,tag):
    torch.manual_seed(seed); np.random.seed(seed)
    m=make(); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); lf=nn.CrossEntropyLoss()
    rng=np.random.default_rng(seed)
    for ep in range(epochs):
        I = render.render(F.rotate(Ptr,0,jitter=True,rng=rng)) if aug else Itr0
        m.train(); tot=c=n=0
        for xb,yb in loader(I,ytr,seed=seed*100+ep):
            opt.zero_grad(); z=m(xb); l=lf(z,yb); l.backward(); opt.step()
            tot+=l.item()*len(yb); c+=(z.argmax(1)==yb).sum().item(); n+=len(yb)
        sch.step()
        print(f'  {tag} seed{seed} ep{ep+1}/{epochs} loss {tot/n:.4f} acc {100*c/n:.2f}%',flush=True)
    return m

@torch.no_grad()
def sweep(m,tag):
    m.eval(); res={}
    for a in ANG:
        I=Ite_ang[a]; P=[]
        for i in range(0,len(I),256):
            P.append(m(torch.tensor(I[i:i+256]).unsqueeze(1)).argmax(1).numpy())
        p=np.concatenate(P); k=np.isin(yte,[6,9])
        res[a]=(100*(p==yte).mean(), 100*(p[k]==yte[k]).mean())
        if a==0: res['pred0']=p.tolist()
    ov=[res[a][0] for a in ANG]; s69=[res[a][1] for a in ANG]
    print(f'>> {tag}: overall min/mean {min(ov):.2f}/{np.mean(ov):.2f} | 6&9 min/mean {min(s69):.2f}/{np.mean(s69):.2f}',flush=True)
    return res

out={}
jobs=[('CNN (upright train)',lambda:CNN(),False,15,[0,1,2]),
      ('CNN + rot. augmentation',lambda:CNN(),True,15,[0,1,2]),
      ('RIC-CNN (upright train)',lambda:RICCNN(),False,12,[0,1]),
      ('TI-pooling-8 (upright train)',lambda:TIPool(8),False,12,[0,1])]
for name,mk,aug,ep,seeds in jobs:
    for sd in seeds:
        t=time.time(); m=train(mk,aug,ep,sd,name)
        out.setdefault(name,{})[sd]=sweep(m,f'{name} seed{sd}')
        print(f'   [{name} seed{sd}] {time.time()-t:.0f}s',flush=True)
        json.dump(out,open('img_results.json','w'))
print('DONE')
