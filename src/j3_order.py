"""Journal experiment 3: how much of the trajectory advantage can be recovered
from an offline image? A rotation-augmented CNN is trained to predict the
invariant order-aware descriptor from the rendered image, and the frozen
trajectory classifier is then applied to the predicted descriptor."""
import numpy as np, torch, torch.nn as nn, json, time, feats as F, render
from sklearn.ensemble import ExtraTreesClassifier
torch.set_num_threads(2)
Ptr,ytr,Pte,yte=F.load(); K=np.isin(yte,[6,9]); ANG=list(range(0,360,10))
Dtr=F.f_full2(Ptr); Dte=F.f_full2(Pte)
mu,sd=Dtr.mean(0),Dtr.std(0)+1e-9
et=ExtraTreesClassifier(n_estimators=600,random_state=0,n_jobs=2).fit(Dtr,ytr)   # frozen, trained on TRUE descriptors
print('descriptor dim',Dtr.shape[1],flush=True)
Ite={a:render.render(F.rotate(Pte,a)) for a in ANG}
def net(out):
    return nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(64,128,3,padding=1),nn.ReLU(),
                         nn.AdaptiveAvgPool2d(2),nn.Flatten(),
                         nn.Linear(512,256),nn.ReLU(),nn.Linear(256,out))
R={}
for seed in (0,1,2):
    torch.manual_seed(seed); r=np.random.default_rng(seed)
    m=net(Dtr.shape[1]); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4)
    epochs=25; sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); lf=nn.SmoothL1Loss()
    T=torch.tensor(((Dtr-mu)/sd).astype(np.float32))
    for ep in range(epochs):
        I=render.render(F.rotate(Ptr,0,jitter=True,rng=r))    # rotation augmented input, invariant target
        o=r.permutation(len(ytr)); m.train(); tot=0.
        for i in range(0,len(o),128):
            j=o[i:i+128]
            opt.zero_grad(); z=m(torch.tensor(I[j]).unsqueeze(1)); l=lf(z,T[j]); l.backward(); opt.step()
            tot+=l.item()*len(j)
        sch.step()
        if (ep+1)%5==0: print(f'   seed{seed} ep{ep+1}/{epochs} loss {tot/len(o):.4f}',flush=True)
    m.eval(); ov=[];s69=[];sc=[]
    with torch.no_grad():
        for a in ANG:
            I=Ite[a]
            P=np.concatenate([m(torch.tensor(I[i:i+256]).unsqueeze(1)).numpy() for i in range(0,len(I),256)])
            D=P*sd+mu
            p=et.predict(D); ov.append(100*(p==yte).mean()); s69.append(100*(p[K]==yte[K]).mean())
    print(f'>> predicted descriptor + frozen classifier, seed{seed}: overall {np.mean(ov):.2f} (worst {min(ov):.2f}) | 6&9 {np.mean(s69):.2f} (worst {min(s69):.2f})',flush=True)
    R[seed]=dict(ov=ov,s69=s69)
    json.dump(R,open('j3_order_results.json','w'))
print('J3 DONE',flush=True)
