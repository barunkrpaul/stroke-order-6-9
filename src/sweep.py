import numpy as np, feats as F
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
Ptr,ytr,Pte,yte=F.load()
itr,iva=train_test_split(np.arange(len(ytr)),test_size=0.2,random_state=0,stratify=ytr)
fsets={'turn13':F.f_turn,'multiscale':F.f_multiscale,'dist':F.f_dist,'full':F.f_full,'raw':F.f_raw}
models={
 'ET400':lambda:ExtraTreesClassifier(n_estimators=400,random_state=0,n_jobs=2),
 'HGB':lambda:HistGradientBoostingClassifier(random_state=0),
 'MLP':lambda:make_pipeline(StandardScaler(),MLPClassifier((256,128),max_iter=600,random_state=0)),
}
print(f"{'features':<12}{'model':<7}{'val':>8}{'val 6/9':>9}")
for fn_name,fn in fsets.items():
    Xtr=fn(Ptr)
    for mn,mk in models.items():
        m=mk().fit(Xtr[itr],ytr[itr]); p=m.predict(Xtr[iva])
        m69=np.isin(ytr[iva],[6,9])
        print(f"{fn_name:<12}{mn:<7}{100*accuracy_score(ytr[iva],p):>7.2f}%{100*accuracy_score(ytr[iva][m69],p[m69]):>8.2f}%")
