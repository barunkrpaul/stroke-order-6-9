"""J1b: the rotation-augmented CNN on MNIST at a proper training budget
(all 60000 training images, 15 epochs), so that the ceiling on the pair cannot
be attributed to undertraining."""
import numpy as np, torch, torch.nn as nn, json, time
from scipy.ndimage import rotate as ndrot
torch.set_num_threads(2)
X=np.load('mnist_X.npy').reshape(-1,28,28).astype(np.float32)/255.
y=np.load('mnist_y.npy')
rng=np.random.default_rng(0); idx=rng.permutation(len(y)); X,y=X[idx],y[idx]
Xtr,ytr,Xte,yte=X[:60000],y[:60000],X[60000:],y[60000:]
K=np.isin(yte,[6,9]); ANG=list(range(0,360,30))
TEST={a:np.stack([ndrot(im,a,reshape=False,order=1) for im in Xte]) for a in ANG}
def net():
    return nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(64,128,3,padding=1),nn.ReLU(),
                         nn.AdaptiveAvgPool2d(2),nn.Flatten(),nn.Linear(512,10))
R={}
for sd in (0,1):
    torch.manual_seed(sd); r=np.random.default_rng(100+sd); m=net()
    opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4); ep_n=15
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,ep_n); lf=nn.CrossEntropyLoss()
    for ep in range(ep_n):
        t0=time.time()
        I=np.stack([ndrot(im,a,reshape=False,order=1) for im,a in zip(Xtr,r.uniform(0,360,len(Xtr)))])
        o=r.permutation(len(ytr)); m.train(); c=n=0
        for i in range(0,len(o),256):
            j=o[i:i+256]
            xb=torch.tensor(I[j]).unsqueeze(1); yb=torch.tensor(ytr[j])
            opt.zero_grad(); z=m(xb); l=lf(z,yb); l.backward(); opt.step()
            c+=(z.argmax(1)==yb).sum().item(); n+=len(yb)
        sch.step(); print(f'  seed{sd} ep{ep+1}/{ep_n} train {100*c/n:.2f}% ({time.time()-t0:.0f}s)',flush=True)
    m.eval(); ov=[];s69=[]
    with torch.no_grad():
        for a in ANG:
            I=TEST[a]
            p=np.concatenate([m(torch.tensor(I[i:i+512]).unsqueeze(1)).argmax(1).numpy() for i in range(0,len(I),512)])
            ov.append(100*(p==yte).mean()); s69.append(100*(p[K]==yte[K]).mean())
            if a==0: cm=[[int(((yte==t)&(p==q)).sum()) for q in (6,9)] for t in (6,9)]
    print(f'>> MNIST CNN aug (60k, 15 ep) seed{sd}: overall {np.mean(ov):.2f} (worst {min(ov):.2f}) | 6&9 {np.mean(s69):.2f} (worst {min(s69):.2f}) cm@0 {cm}',flush=True)
    R[sd]=dict(ov=ov,s69=s69,cm=cm)
    json.dump(R,open('j1b_mnist_results.json','w'))
print('J1b DONE',flush=True)
