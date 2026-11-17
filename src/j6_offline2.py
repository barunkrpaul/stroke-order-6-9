"""J6: two better-posed versions of the offline recovery.
(a) the network predicts the single order-dependent scalar (accumulated turning)
    and the threshold rule of the paper is applied to the prediction;
(b) the tree ensemble is refitted on predicted descriptors, which removes the
    distribution shift that breaks the frozen classifier of J3."""
import numpy as np, torch, torch.nn as nn, json, time, feats as F, render
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
torch.set_num_threads(2)
Ptr,ytr,Pte,yte=F.load(); K=np.isin(yte,[6,9]); ANG=list(range(0,360,30))
THR=json.load(open('scalar_rule.json'))['thr']
tot=lambda P: F.turning(P)[0].sum(1)
Str,Ste=tot(Ptr),tot(Pte)
Dtr=F.f_full2(Ptr); mu,sd=Dtr.mean(0),Dtr.std(0)+1e-9
Ite={a:render.render(F.rotate(Pte,a)) for a in ANG}
def net(out):
    return nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(64,128,3,padding=1),nn.ReLU(),
                         nn.AdaptiveAvgPool2d(2),nn.Flatten(),
                         nn.Linear(512,256),nn.ReLU(),nn.Linear(256,out))
def train(target,out,seed,epochs=12):
    torch.manual_seed(seed); r=np.random.default_rng(seed)
    m=net(out); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); lf=nn.SmoothL1Loss()
    T=torch.tensor(target.astype(np.float32))
    for ep in range(epochs):
        I=render.render(F.rotate(Ptr,0,jitter=True,rng=r)); o=r.permutation(len(ytr)); m.train()
        for i in range(0,len(o),128):
            j=o[i:i+128]
            opt.zero_grad(); z=m(torch.tensor(I[j]).unsqueeze(1))
            l=lf(z,T[j].view(len(j),-1)); l.backward(); opt.step()
        sch.step()
    return m
@torch.no_grad()
def predict(m,I,out):
    return np.concatenate([m(torch.tensor(I[i:i+256]).unsqueeze(1)).numpy() for i in range(0,len(I),256)]).reshape(len(I),out)

R={}
# (a) scalar
acc=[];mus=[]
for sd_ in (0,1):
    m=train(Str[:,None],1,sd_)
    a69=[]
    for a in ANG:
        p=predict(m,Ite[a],1)[:,0]
        lab=np.where(p>THR,6,9); a69.append(100*(lab[K]==yte[K]).mean())
    p0=predict(m,Ite[0],1)[:,0]
    mus.append((float(p0[yte==6].mean()),float(p0[yte==6].std()),float(p0[yte==9].mean()),float(p0[yte==9].std())))
    acc.append(float(np.mean(a69)))
    print(f'>> scalar predicted from the image, seed{sd_}: 6&9 {np.mean(a69):.2f} (worst {min(a69):.2f}) | '
          f'predicted mean for 6 {mus[-1][0]:.2f}+-{mus[-1][1]:.2f}, for 9 {mus[-1][2]:.2f}+-{mus[-1][3]:.2f} '
          f'(true 6.16 and 0.93)',flush=True)
R['scalar']=dict(acc69=acc,dist=mus,thr=THR)
json.dump(R,open('j6_offline2_results.json','w'))

# (b) descriptor with a refitted classifier
itr,iva=train_test_split(np.arange(len(ytr)),test_size=0.25,random_state=0,stratify=ytr)
ov=[];a69=[]
for sd_ in (0,1):
    m=train(((Dtr-mu)/sd),Dtr.shape[1],sd_)
    r=np.random.default_rng(50+sd_)
    Iva=render.render(F.rotate(Ptr[iva],0,jitter=True,rng=r))
    Dva=predict(m,Iva,Dtr.shape[1])*sd+mu
    et=ExtraTreesClassifier(n_estimators=600,random_state=0,n_jobs=2).fit(Dva,ytr[iva])  # fitted on PREDICTED descriptors
    o=[];s=[]
    for a in ANG:
        D=predict(m,Ite[a],Dtr.shape[1])*sd+mu
        p=et.predict(D); o.append(100*(p==yte).mean()); s.append(100*(p[K]==yte[K]).mean())
    ov.append(float(np.mean(o))); a69.append(float(np.mean(s)))
    print(f'>> descriptor predicted, classifier refitted, seed{sd_}: overall {np.mean(o):.2f} (worst {min(o):.2f}) | '
          f'6&9 {np.mean(s):.2f} (worst {min(s):.2f})',flush=True)
R['refit']=dict(ov=ov,acc69=a69)
json.dump(R,open('j6_offline2_results.json','w'))
print('J6 DONE',flush=True)
