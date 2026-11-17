"""Journal experiment 4: robustness to jitter, to missing points and to a wrong
stroke order, behaviour under reflection, and the computational cost."""
import numpy as np, json, time, feats as F
from sklearn.ensemble import ExtraTreesClassifier
Ptr,ytr,Pte,yte=F.load(); K=np.isin(yte,[6,9]); R={}
et  =ExtraTreesClassifier(n_estimators=600,random_state=0,n_jobs=2).fit(F.f_full2(Ptr),ytr)
etns=ExtraTreesClassifier(n_estimators=600,random_state=0,n_jobs=2).fit(F.f_full2_nosign(Ptr),ytr)
def ev(P,fn=F.f_full2,model=None):
    p=(model or et).predict(fn(P)); return 100*(p==yte).mean(),100*(p[K]==yte[K]).mean()
base=ev(Pte); print('clean: overall %.2f | 6&9 %.2f'%base,flush=True)
R['clean']=base

rng=np.random.default_rng(0)
R['jitter']={}
for s in (0.5,1,2,4):
    a=[ev(Pte+rng.normal(0,s,Pte.shape)) for _ in range(3)]
    R['jitter'][s]=(float(np.mean([x[0] for x in a])),float(np.mean([x[1] for x in a])))
    print('jitter sd %.1f (coords are 0-100): overall %.2f | 6&9 %.2f'%(s,*R['jitter'][s]),flush=True)

def resample8(P):
    out=np.empty_like(P[:, :8])
    for i,p in enumerate(P):
        d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        if d[-1]<1e-9: out[i]=p[0]; continue
        t=d/d[-1]; g=np.linspace(0,1,8)
        out[i]=np.stack([np.interp(g,t,p[:,0]),np.interp(g,t,p[:,1])],1)
    return out
R['dropped']={}
for nd in (1,2):
    acc=[]
    for rep in range(3):
        Q=[]
        for p in Pte:
            keep=np.ones(8,bool); keep[rng.choice(np.arange(1,7),nd,replace=False)]=False
            Q.append(p[keep])
        Q=resample8(np.array([np.pad(q,((0,8-len(q)),(0,0)),mode='edge') if len(q)<8 else q for q in Q]))
        acc.append(ev(Q))
    R['dropped'][nd]=(float(np.mean([x[0] for x in acc])),float(np.mean([x[1] for x in acc])))
    print('%d interior point(s) dropped and re-resampled: overall %.2f | 6&9 %.2f'%(nd,*R['dropped'][nd]),flush=True)

R['wrong_order']={}
for frac in (0.1,0.25,0.5,1.0):
    Q=Pte.copy(); sel=rng.random(len(Q))<frac; Q[sel]=Q[sel][:,::-1,:]
    R['wrong_order'][frac]=ev(Q)
    print('stroke order reversed for %.0f%% of test samples: overall %.2f | 6&9 %.2f'%(100*frac,*R['wrong_order'][frac]),flush=True)

M=Pte.copy(); M[...,0]=-M[...,0]                      # reflection
R['mirror_ours']=ev(M); R['mirror_nosign']=ev(M,F.f_full2_nosign,etns)
R['clean_nosign']=ev(Pte,F.f_full2_nosign,etns)
print('mirrored input, signed descriptor: overall %.2f | 6&9 %.2f'%R['mirror_ours'],flush=True)
print('mirrored input, unsigned variant : overall %.2f | 6&9 %.2f (clean %.2f / %.2f)'
      %(*R['mirror_nosign'],*R['clean_nosign']),flush=True)

t=time.time(); F.f_full2(Pte); t_desc=(time.time()-t)/len(Pte)*1e6
t=time.time(); ExtraTreesClassifier(n_estimators=600,random_state=1,n_jobs=2).fit(F.f_full2(Ptr),ytr); t_fit=time.time()-t
t=time.time(); et.predict(F.f_full2(Pte)); t_pred=(time.time()-t)/len(Pte)*1e6
R['cost']=dict(desc_us_per_sample=t_desc,fit_s=t_fit,predict_us_per_sample=t_pred)
print('cost: descriptor %.0f us/sample, training %.1f s, prediction %.0f us/sample (2 CPU cores, no GPU)'
      %(t_desc,t_fit,t_pred),flush=True)
json.dump(R,open('j4_robust_results.json','w'))
print('J4 DONE',flush=True)
