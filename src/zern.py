"""Classical exactly-invariant STATIC image descriptors on the same rendered digits:
Zernike magnitudes + Hu moments + radial profile. Shows the 6/9 ceiling of any
orientation-blind representation of the image."""
import numpy as np, json, mahotas, feats as F, render
from skimage.measure import moments_central, moments_normalized, moments_hu
from sklearn.ensemble import ExtraTreesClassifier
Ptr,ytr,Pte,yte=F.load()
def desc(I):
    out=[]
    for im in I:
        z=mahotas.features.zernike_moments(im,radius=14,degree=12)
        hu=moments_hu(moments_normalized(moments_central(im)))
        c=np.array(im.shape)/2.
        yy,xx=np.mgrid[:im.shape[0],:im.shape[1]]
        r=np.sqrt((yy-c[0])**2+(xx-c[1])**2)
        rad=np.array([im[(r>=a)&(r<a+2)].sum() for a in range(0,16,2)])
        rad=rad/(rad.sum()+1e-9)
        out.append(np.concatenate([z,np.sign(hu)*np.log1p(np.abs(hu)),rad]))
    return np.array(out)
Xtr=desc(render.render(Ptr))
res={}
m=ExtraTreesClassifier(n_estimators=600,random_state=0,n_jobs=2).fit(Xtr,ytr)
for a in range(0,360,10):
    p=m.predict(desc(render.render(F.rotate(Pte,a)))); k=np.isin(yte,[6,9])
    res[a]=(100*(p==yte).mean(),100*(p[k]==yte[k]).mean())
    if a==0: res['pred0']=p.tolist()
ov=[res[a][0] for a in range(0,360,10)]; s=[res[a][1] for a in range(0,360,10)]
print(f'Zernike+Hu+radial (static invariant): overall {np.mean(ov):.2f} (min {min(ov):.2f}) | 6&9 {np.mean(s):.2f} (min {min(s):.2f})')
json.dump(res,open('zern_results.json','w'))
