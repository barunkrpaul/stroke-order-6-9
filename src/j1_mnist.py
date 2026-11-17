"""Journal experiment 1: the 6/9 ceiling on real offline bitmaps (full MNIST).
(a) CNN trained upright, (b) CNN with full-circle rotation augmentation,
(c) exactly rotation-invariant static descriptor (Zernike + Hu + radial)."""
import numpy as np, torch, torch.nn as nn, json, time
from scipy.ndimage import rotate as ndrot
torch.set_num_threads(2)
X=np.load('mnist_X.npy').reshape(-1,28,28).astype(np.float32)/255.
y=np.load('mnist_y.npy')
rng=np.random.default_rng(0); idx=rng.permutation(len(y))
X,y=X[idx],y[idx]
NTR=20000  # comparable to the 12000 images of the standard MNIST-rot protocol
Xtr,ytr,Xte,yte=X[:NTR],y[:NTR],X[60000:],y[60000:]
K=np.isin(yte,[6,9]); ANG=list(range(0,360,30))
print('train',Xtr.shape,'test',Xte.shape,'pair samples',int(K.sum()),flush=True)

def rot(A,ang):        return np.stack([ndrot(a,ang,reshape=False,order=1) for a in A])
def rot_rand(A,rng):   return np.stack([ndrot(a,b,reshape=False,order=1) for a,b in zip(A,rng.uniform(0,360,len(A)))])

TEST={}
t=time.time()
for a in ANG: TEST[a]=rot(Xte,a)
print('rotated test sets built in %.0f s'%(time.time()-t),flush=True)

def net():
    return nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(64,128,3,padding=1),nn.ReLU(),
                         nn.AdaptiveAvgPool2d(2),nn.Flatten(),nn.Linear(512,10))
@torch.no_grad()
def sweep(m,tag):
    m.eval(); ov=[];s69=[]
    for a in ANG:
        I=TEST[a]; P=[]
        for i in range(0,len(I),512):
            P.append(m(torch.tensor(I[i:i+512]).unsqueeze(1)).argmax(1).numpy())
        p=np.concatenate(P); ov.append(100*(p==yte).mean()); s69.append(100*(p[K]==yte[K]).mean())
    print(f'>> {tag}: overall mean {np.mean(ov):.2f} worst {min(ov):.2f} | 6&9 mean {np.mean(s69):.2f} worst {min(s69):.2f}',flush=True)
    return ov,s69

def train(aug,seed,epochs):
    torch.manual_seed(seed); r=np.random.default_rng(seed); m=net()
    opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs); lf=nn.CrossEntropyLoss()
    for ep in range(epochs):
        I=rot_rand(Xtr,r) if aug else Xtr
        o=r.permutation(len(ytr)); m.train(); c=n=0; t0=time.time()
        for i in range(0,len(o),256):
            j=o[i:i+256]
            xb=torch.tensor(I[j]).unsqueeze(1); yb=torch.tensor(ytr[j])
            opt.zero_grad(); z=m(xb); l=lf(z,yb); l.backward(); opt.step()
            c+=(z.argmax(1)==yb).sum().item(); n+=len(yb)
        sch.step(); print(f'   {"aug" if aug else "upright"} s{seed} ep{ep+1}/{epochs} train {100*c/n:.2f}% ({time.time()-t0:.0f}s)',flush=True)
    return m

R={}
R['cnn_upright']=[sweep(train(False,0,6),'MNIST CNN upright-trained')]
R['cnn_aug']=[sweep(train(True,s,6),f'MNIST CNN rotation-augmented seed{s}') for s in (0,1)]
json.dump(R,open('j1_mnist_results.json','w'))

# ---- exactly invariant static descriptor ----
import mahotas
from skimage.measure import moments_central, moments_normalized, moments_hu
from sklearn.ensemble import ExtraTreesClassifier
def desc(I):
    out=[]
    for im in I:
        z=mahotas.features.zernike_moments(im,radius=13,degree=12)
        hu=moments_hu(moments_normalized(moments_central(im)))
        c=np.array(im.shape)/2.; yy,xx=np.mgrid[:28,:28]
        r=np.sqrt((yy-c[0])**2+(xx-c[1])**2)
        rad=np.array([im[(r>=a)&(r<a+2)].sum() for a in range(0,14,2)]); rad=rad/(rad.sum()+1e-9)
        out.append(np.concatenate([z,np.sign(hu)*np.log1p(np.abs(hu)),rad]))
    return np.array(out)
NSUB=10000; t=time.time()
Ftr=desc(Xtr[:NSUB]); print('train descriptors %.0f s'%(time.time()-t),flush=True)
et=ExtraTreesClassifier(n_estimators=600,random_state=0,n_jobs=2).fit(Ftr,ytr[:NSUB])
ov=[];s69=[]
for a in ANG:
    p=et.predict(desc(TEST[a])); ov.append(100*(p==yte).mean()); s69.append(100*(p[K]==yte[K]).mean())
    print(f'   static invariant at {a:3d} deg: overall {ov[-1]:.2f} | 6&9 {s69[-1]:.2f}',flush=True)
print(f'>> MNIST static invariant (Zernike+Hu+radial): overall mean {np.mean(ov):.2f} | 6&9 mean {np.mean(s69):.2f}',flush=True)
R['static_inv']=[(ov,s69)]
json.dump(R,open('j1_mnist_results.json','w'))
print('J1 DONE',flush=True)
