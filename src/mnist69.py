"""Control on real offline bitmaps: MNIST 6s and 9s under full 360 deg rotation.
Shows the same ~50% ceiling that the trajectory model escapes."""
import json, numpy as np, torch, torch.nn as nn
from scipy.ndimage import rotate as ndrot
torch.set_num_threads(2)
def load(d):
    v=json.load(open(f'npmm/package/src/digits/{d}.json'))['data']
    return np.array(v,dtype=np.float32).reshape(-1,28,28)
X=[];y=[]
for d in range(10):
    a=load(d); X.append(a); y+=[d]*len(a)
X=np.concatenate(X); y=np.array(y)
rng=np.random.default_rng(0); idx=rng.permutation(len(y))
X,y=X[idx],y[idx]; ntr=int(.8*len(y))
def rot_batch(A,rng): return np.stack([ndrot(a,rng.uniform(0,360),reshape=False,order=1) for a in A])
Xtr,ytr,Xte,yte=X[:ntr],y[:ntr],X[ntr:],y[ntr:]
print('train',Xtr.shape,'test',Xte.shape)
def net():
    return nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(64,128,3,padding=1),nn.ReLU(),
                         nn.AdaptiveAvgPool2d(2),nn.Flatten(),nn.Linear(512,10))
res={}
for sd in (0,1):
    torch.manual_seed(sd); rng=np.random.default_rng(100+sd)
    m=net(); opt=torch.optim.Adam(m.parameters(),1e-3); lf=nn.CrossEntropyLoss()
    for ep in range(12):
        Xa=rot_batch(Xtr,rng); p=rng.permutation(len(ytr)); m.train(); c=n=0
        for i in range(0,len(p),128):
            j=p[i:i+128]; xb=torch.tensor(Xa[j]).unsqueeze(1); yb=torch.tensor(ytr[j])
            opt.zero_grad(); z=m(xb); l=lf(z,yb); l.backward(); opt.step()
            c+=(z.argmax(1)==yb).sum().item(); n+=len(yb)
        print(f'  seed{sd} ep{ep+1} train acc {100*c/n:.2f}%',flush=True)
    m.eval(); Xr=rot_batch(Xte,np.random.default_rng(7)); P=[]
    with torch.no_grad():
        for i in range(0,len(Xr),256): P.append(m(torch.tensor(Xr[i:i+256]).unsqueeze(1)).argmax(1).numpy())
    p=np.concatenate(P); k=np.isin(yte,[6,9])
    ov=100*(p==yte).mean(); s69=100*(p[k]==yte[k]).mean()
    print(f'>> MNIST-rot CNN+aug seed{sd}: overall {ov:.2f}% | 6&9 {s69:.2f}%',flush=True)
    res[sd]=dict(overall=float(ov),acc69=float(s69),
                 cm69=[[int(((yte==t)&(p==q)).sum()) for q in (6,9)] for t in (6,9)])
json.dump(res,open('mnist69_results.json','w'))
