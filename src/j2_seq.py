"""Journal experiment 2: sequence models over the ordered turning-angle profile.
The 8 pen positions are interpolated by a chord-length cubic spline and resampled
to L points, which gives a smooth rotation-invariant curvature profile."""
import numpy as np, torch, torch.nn as nn, json, time, feats as F
from scipy.interpolate import CubicSpline
from sklearn.ensemble import ExtraTreesClassifier
torch.set_num_threads(2)
Ptr,ytr,Pte,yte=F.load(); K=np.isin(yte,[6,9]); L=32

def spline_resample(P,L=L):
    out=np.empty((len(P),L,2))
    for i,p in enumerate(P):
        d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        if d[-1]<1e-9: out[i]=p[0]; continue
        t=d/d[-1]; u,ix=np.unique(t,return_index=True)
        cs=CubicSpline(u,p[ix],axis=0)
        out[i]=cs(np.linspace(0,1,L))
    return out

def seq_feats(P):
    """(sin k, cos k, normalised step length) at every interior point: invariant."""
    Q=spline_resample(P)
    v=np.diff(Q,axis=1); Ln=np.linalg.norm(v,axis=-1)
    u=v/(Ln[...,None]+1e-9)
    cr=u[:,:-1,0]*u[:,1:,1]-u[:,:-1,1]*u[:,1:,0]
    dt=(u[:,:-1]*u[:,1:]).sum(-1)
    k=np.arctan2(cr,dt)
    ln=Ln[:,1:]/(Ln.sum(1,keepdims=True)+1e-9)
    return np.stack([np.sin(k),np.cos(k),ln],1).astype(np.float32)   # (N,3,L-2)

t=time.time(); Str,Ste=seq_feats(Ptr),seq_feats(Pte)
print('sequence features',Str.shape,'built in %.0f s'%(time.time()-t),flush=True)
dev=np.abs(seq_feats(F.rotate(Pte,137))-Ste).max()
print('max |sequence feature change| under a 137 deg rotation: %.2e'%dev,flush=True)

class CNN1D(nn.Module):
    def __init__(s):
        super().__init__()
        s.f=nn.Sequential(nn.Conv1d(3,64,5,padding=2),nn.ReLU(),nn.Conv1d(64,128,5,padding=2),
                          nn.ReLU(),nn.AdaptiveMaxPool1d(1),nn.Flatten())
        s.c=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,10))
    def forward(s,x): return s.c(s.f(x))

class BiLSTM(nn.Module):
    def __init__(s,h=64):
        super().__init__(); s.r=nn.LSTM(3,h,batch_first=True,bidirectional=True); s.c=nn.Linear(2*h,10)
    def forward(s,x):
        o,_=s.r(x.transpose(1,2)); return s.c(o.mean(1))

def run(make,name,epochs=30,seeds=(0,1,2)):
    accs=[];a69=[]
    for sd in seeds:
        torch.manual_seed(sd); r=np.random.default_rng(sd); m=make()
        opt=torch.optim.Adam(m.parameters(),2e-3,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); lf=nn.CrossEntropyLoss()
        for ep in range(epochs):
            o=r.permutation(len(ytr)); m.train()
            for i in range(0,len(o),128):
                j=o[i:i+128]
                opt.zero_grad(); z=m(torch.tensor(Str[j])); l=lf(z,torch.tensor(ytr[j]))
                l.backward(); opt.step()
            sch.step()
        m.eval()
        with torch.no_grad():
            p=np.concatenate([m(torch.tensor(Ste[i:i+512])).argmax(1).numpy() for i in range(0,len(Ste),512)])
        accs.append(100*(p==yte).mean()); a69.append(100*(p[K]==yte[K]).mean())
        print(f'   {name} seed{sd}: overall {accs[-1]:.2f} | 6&9 {a69[-1]:.2f}',flush=True)
    print(f'>> {name}: overall {np.mean(accs):.2f}+-{np.std(accs):.2f} | 6&9 {np.mean(a69):.2f}+-{np.std(a69):.2f}',flush=True)
    return dict(acc=accs,acc69=a69)

R={}
R['cnn1d']=run(CNN1D,'1-D CNN on turning profile')
R['bilstm']=run(BiLSTM,'BiLSTM on turning profile')
# trees on the same flattened sequence, for a like-for-like comparison
Xtr=Str.reshape(len(Str),-1); Xte=Ste.reshape(len(Ste),-1)
acc=[];a69=[]
for sd in range(3):
    et=ExtraTreesClassifier(n_estimators=600,random_state=sd,n_jobs=2).fit(Xtr,ytr)
    p=et.predict(Xte); acc.append(100*(p==yte).mean()); a69.append(100*(p[K]==yte[K]).mean())
print(f">> trees on the same sequence: overall {np.mean(acc):.2f}+-{np.std(acc):.2f} | 6&9 {np.mean(a69):.2f}+-{np.std(a69):.2f}",flush=True)
R['trees_seq']=dict(acc=acc,acc69=a69)
R['invariance_dev']=float(dev)
json.dump(R,open('j2_seq_results.json','w'))
print('J2 DONE',flush=True)
