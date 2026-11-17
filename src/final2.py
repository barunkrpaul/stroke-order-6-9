"""Wrap-free descriptor: exact invariance + 5 seeds + ablations + McNemar."""
import numpy as np, json, feats as F
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from scipy.stats import binomtest
Ptr,ytr,Pte,yte=F.load(); ANG=list(range(0,360,10)); K=np.isin(yte,[6,9])
base=F.f_full2(Pte)
print('max |feature change| over 36 rotations: %.2e'%max(np.abs(F.f_full2(F.rotate(Pte,a))-base).max() for a in ANG),flush=True)
R={}
for name,fn in [('ours',F.f_full2),('nosign',F.f_full2_nosign),('noorder',F.f_full2_noorder)]:
    Xtr=fn(Ptr); acc=[];a69=[];pred=None
    for sd in range(5):
        m=ExtraTreesClassifier(n_estimators=600,random_state=sd,n_jobs=2).fit(Xtr,ytr)
        p=m.predict(fn(Pte))
        if sd==0:
            pred=p
            ident=all((m.predict(fn(F.rotate(Pte,a)))==p).all() for a in ANG)
            print(f'  [{name}] identical predictions at all 36 angles: {ident}',flush=True)
        acc.append(100*(p==yte).mean()); a69.append(100*(p[K]==yte[K]).mean())
    R[name]=dict(acc=acc,acc69=a69,pred0=pred.tolist(),
                 cm69=confusion_matrix(yte[K],pred[K],labels=[6,9]).tolist(),
                 nfeat=int(Xtr.shape[1]))
    print(f'{name:<8} feats {Xtr.shape[1]:>3}  overall {np.mean(acc):.2f}+-{np.std(acc):.2f}  '
          f'6&9 {np.mean(a69):.2f}+-{np.std(a69):.2f}  cm {R[name]["cm69"]}',flush=True)
def mc(a,b,lab):
    a,b=np.array(a),np.array(b); oa=a[K]==yte[K]; ob=b[K]==yte[K]
    n10=int((oa&~ob).sum()); n01=int((~oa&ob).sum())
    p=binomtest(n10,n10+n01,0.5).pvalue if n10+n01 else 1.0
    print(f'  {lab}: ours correct/other wrong {n10}, ours wrong/other correct {n01}, p={p:.3g}',flush=True)
    return dict(n10=n10,n01=n01,p=float(p))
print('McNemar on digits 6 and 9:')
R['mc_noorder']=mc(R['ours']['pred0'],R['noorder']['pred0'],'ours vs stroke-direction-unknown')
R['mc_nosign'] =mc(R['ours']['pred0'],R['nosign']['pred0'], 'ours vs signs-removed')
json.dump(R,open('final2_results.json','w'))
