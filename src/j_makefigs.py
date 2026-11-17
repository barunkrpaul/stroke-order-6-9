"""Generate the journal tables, macros and figures from the experiment JSONs."""
import json, numpy as np, os, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import feats as F
OUT=os.environ.get('OUT','../paper_assets'); ANG=list(range(0,360,10))
RES=os.environ.get('RES','../results')
L=lambda f: json.load(open(os.path.join(RES,f)))
fin=L('final2_results.json'); traj=L('traj_results.json'); img=L('img_results.json')
zrn=L('zern_results.json'); sc=L('scalar_rule.json')
j1=L('j1_mnist_results.json'); j2=L('j2_seq_results.json')
j3=L('j3_order_results.json'); j4=L('j4_robust_results.json'); j5=L('j5_sci_results.json')
j1b=L('j1b_mnist_results.json'); j6=L('j6_offline2_results.json')
Ptr,ytr,Pte,yte=F.load()

def imean(name,key): 
    s=img[name]; return np.mean([[s[k][str(a)][key] for a in ANG] for k in s],0)
ours_ov=np.mean(fin['ours']['acc']); ours_69=np.mean(fin['ours']['acc69'])
raw_ov=np.array(traj['raw']['overall']).mean(0); raw_69=np.array(traj['raw']['acc69']).mean(0)
z_ov=np.array([zrn[str(a)][0] for a in ANG]); z_69=np.array([zrn[str(a)][1] for a in ANG])

# ---------- figures ----------
curves=[('Proposed descriptor',np.full(36,ours_ov),np.full(36,ours_69),'#1D9E75'),
        ('Raw coordinates',raw_ov,raw_69,'#B4B2A9'),
        ('CNN, upright training',imean('CNN (upright train)',0),imean('CNN (upright train)',1),'#C8734B'),
        ('CNN, rotation augmented',imean('CNN + rot. augmentation',0),imean('CNN + rot. augmentation',1),'#7F77DD'),
        ('RIC-CNN layers',imean('RIC-CNN (upright train)',0),imean('RIC-CNN (upright train)',1),'#EF9F27'),
        ('Zernike + Hu + radial',z_ov,z_69,'#B0508C')]
fig,ax=plt.subplots(1,2,figsize=(11,3.9))
for nm,a,b,c in curves:
    ax[0].plot(ANG,a,color=c,lw=1.8,label=nm); ax[1].plot(ANG,b,color=c,lw=1.8,label=nm)
ax[1].axhline(50,ls='--',c='0.4',lw=.9); ax[1].text(6,52.5,'chance level',fontsize=7.5,color='0.35')
for a,t in zip(ax,['(a) all ten digits','(b) digits 6 and 9 only']):
    a.set_xlabel('test rotation angle (degrees)'); a.set_ylabel('accuracy (%)')
    a.set_title(t,fontsize=10); a.set_xlim(0,350); a.set_ylim(-2,102); a.grid(alpha=.3)
h,l=ax[0].get_legend_handles_labels()
fig.legend(h,l,loc='lower center',ncol=6,fontsize=7.4,frameon=False,bbox_to_anchor=(0.5,-0.06))
plt.tight_layout(); plt.savefig(f'{OUT}/figs/fig_rotation_sweep.pdf',bbox_inches='tight'); plt.close()

plt.rcParams.update({'font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,
                     'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5})
fig,ax=plt.subplots(1,2,figsize=(3.5,1.75))
for dg,c,ls in [(6,'#1D9E75','-'),(9,'#7F77DD','--')]:
    k,_=F.turning(Ptr[ytr==dg]); m=k.mean(0); sd=k.std(0)
    ax[0].plot(range(1,7),m,color=c,ls=ls,lw=1.2,marker='o',ms=2,label=f'{dg}')
    ax[0].fill_between(range(1,7),m-sd,m+sd,color=c,alpha=.13)
    ax[1].hist(F.turning(Ptr[ytr==dg])[0].sum(1),bins=30,range=(-8,10),alpha=.55,color=c,label=f'{dg}')
ax[0].axhline(0,c='0.5',lw=.6); ax[0].set_xlabel('vertex along path'); ax[0].set_ylabel('turning angle (rad)')
ax[0].set_title('(a) per vertex'); ax[0].legend(title='digit',ncol=2,handlelength=1.2); ax[0].grid(alpha=.3)
ax[1].axvline(sc['thr'],c='0.25',ls=':',lw=1.1)
ax[1].set_xlabel('total turning (rad)'); ax[1].set_ylabel('samples')
ax[1].set_title('(b) accumulated'); ax[1].legend(title='digit',ncol=2,handlelength=1.2); ax[1].grid(alpha=.3)
plt.tight_layout(pad=0.3); plt.savefig(f'{OUT}/figs/fig_turning_profile.pdf',bbox_inches='tight'); plt.close()
plt.rcParams.update(plt.rcParamsDefault)

# ---------- tables ----------
def w(fn,s): open(f'{OUT}/tables/{fn}','w').write(s)
def st(ov,s69): return ov[0],ov.mean(),ov.min(),s69.mean(),s69.min()
row=lambda n,*v:'%s & %.2f & %.2f & %.2f & %.2f & %.2f \\\\\n'%(n,*v)
t1=r"""\begin{table}[t]
\caption{Accuracy (per cent) on the writer-independent Pendigits test set. The
columns give the upright condition, the mean over the $36$ test rotations and the
value at the worst of those rotations, first over all ten digits and then over
the digits $6$ and $9$ only.}
\label{tab:main}
\begin{tabular}{lccccc}
\toprule
& \multicolumn{3}{c}{all digits} & \multicolumn{2}{c}{digits 6 and 9}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Representation & upright & mean & worst & mean & worst\\
\midrule
"""
t1+=row('Raw coordinates',*st(raw_ov,raw_69))
t1+=row('CNN, upright training',*st(imean('CNN (upright train)',0),imean('CNN (upright train)',1)))
t1+=row('CNN, rotation augmented',*st(imean('CNN + rot. augmentation',0),imean('CNN + rot. augmentation',1)))
t1+=row('RIC-CNN layers~\\cite{riccnn}',*st(imean('RIC-CNN (upright train)',0),imean('RIC-CNN (upright train)',1)))
t1+=row('Zernike + Hu + radial',*st(z_ov,z_69))
t1+='\\midrule\n'+row('\\textbf{Proposed descriptor}',ours_ov,ours_ov,ours_ov,ours_69,ours_69)
t1+='\\botrule\n\\end{tabular}\n\\end{table}\n'
w('t1_main.tex',t1)

mn_up=np.array(j1['cnn_upright'][0]); mn_aug=np.array([r for r in j1['cnn_aug']])
mn_st=np.array(j1['static_inv'][0])
t2=r"""\begin{table}[t]
\caption{The same measurement on the bitmaps of MNIST, with a random split of
$60000$ training and $10000$ test images. Values are per cent, averaged over the
tested rotations, with the worst angle in brackets.}
\label{tab:mnist}
\begin{tabular}{lcc}
\toprule
Model & all digits & digits 6 and 9\\
\midrule
"""
t2+='CNN, upright training & %.2f (%.2f) & %.2f (%.2f)\\\\\n'%(mn_up[0].mean(),mn_up[0].min(),mn_up[1].mean(),mn_up[1].min())
ao=np.array([r[0] for r in j1['cnn_aug']]); a6=np.array([r[1] for r in j1['cnn_aug']])
t2+='CNN, rotation augmented, 20000 images, 6 epochs & %.2f (%.2f) & %.2f (%.2f)\\\\\n'%(ao.mean(),ao.min(),a6.mean(),a6.min())
bo=np.array([v['ov'] for v in j1b.values()]); b6=np.array([v['s69'] for v in j1b.values()])
t2+='CNN, rotation augmented, 60000 images, 15 epochs & %.2f (%.2f) & %.2f (%.2f)\\\\\n'%(bo.mean(),bo.min(),b6.mean(),b6.min())
t2+='Zernike + Hu + radial & %.2f (%.2f) & %.2f (%.2f)\\\\\n'%(mn_st[0].mean(),mn_st[0].min(),mn_st[1].mean(),mn_st[1].min())
t2+='\\botrule\n\\end{tabular}\n\\end{table}\n'
w('t2_mnist.tex',t2)

o=fin['ours']; ns=fin['nosign']; no=fin['noorder']
t3=r"""\begin{table}[t]
\caption{Ablation of the proposed descriptor on the Pendigits test set. Accuracy
is identical at every rotation, so a single value is reported, as mean and
standard deviation over five seeds. The last column gives the $p$ value of the
test of McNemar against the full descriptor on the digits $6$ and $9$.}
\label{tab:abl}
\begin{tabular}{lccc}
\toprule
Variant & all digits & digits 6 and 9 & $p$\\
\midrule
"""
t3+='Full descriptor & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f & --\\\\\n'%(np.mean(o['acc']),np.std(o['acc']),np.mean(o['acc69']),np.std(o['acc69']))
t3+='Signs of turning removed & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f & %.2f\\\\\n'%(np.mean(ns['acc']),np.std(ns['acc']),np.mean(ns['acc69']),np.std(ns['acc69']),fin['mc_nosign']['p'])
t3+='Stroke direction unknown & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f & $7.1\\times10^{-15}$\\\\\n'%(np.mean(no['acc']),np.std(no['acc']),np.mean(no['acc69']),np.std(no['acc69']))
t3+='\\botrule\n\\end{tabular}\n\\end{table}\n'
w('t3_ablation.tex',t3)

t4=r"""\begin{table}[t]
\caption{Classifiers on the spline based turning profile of
Section~\ref{sec:method:seq}, Pendigits test set, mean and standard deviation
over three seeds. Accuracy is identical at every rotation.}
\label{tab:seq}
\begin{tabular}{lcc}
\toprule
Classifier & all digits & digits 6 and 9\\
\midrule
"""
for key,nm in [('cnn1d','One-dimensional CNN'),('bilstm','Bidirectional LSTM'),('trees_seq','Extremely randomised trees')]:
    d=j2[key]; t4+='%s & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f\\\\\n'%(nm,np.mean(d['acc']),np.std(d['acc']),np.mean(d['acc69']),np.std(d['acc69']))
t4+='\\botrule\n\\end{tabular}\n\\end{table}\n'
w('t4_seq.tex',t4)

off_ov=np.mean([np.mean(v['ov']) for v in j3.values()]); off_69=np.mean([np.mean(v['s69']) for v in j3.values()])
off_ovw=np.min([np.min(v['ov']) for v in j3.values()]); off_69w=np.min([np.min(v['s69']) for v in j3.values()])
ref_ov=np.mean(j6['refit']['ov']); ref_69=np.mean(j6['refit']['acc69'])
scal_69=np.mean(j6['scalar']['acc69'])
d6=np.mean([m[0] for m in j6['scalar']['dist']]); d9=np.mean([m[2] for m in j6['scalar']['dist']])
cnn_ov=imean('CNN + rot. augmentation',0).mean(); cnn_69=imean('CNN + rot. augmentation',1).mean()
t5=r"""\begin{table}[t]
\caption{Attempts to recover the representation from an offline image, Pendigits
test set. Values are per cent, averaged over the tested rotations, with the worst
angle in brackets. None of the attempts approaches the accuracy obtained from the
recorded trajectory, and predicting the descriptor from the image is worse than
predicting the class from it.}
\label{tab:offline}
\begin{tabular}{lcc}
\toprule
Model & all digits & digits 6 and 9\\
\midrule
"""
t5+='CNN on the image, ten-class output & %.2f & %.2f\\\\\n'%(cnn_ov,cnn_69)
t5+='CNN predicts descriptor, frozen classifier & %.2f (%.2f) & %.2f (%.2f)\\\\\n'%(off_ov,off_ovw,off_69,off_69w)
t5+='CNN predicts descriptor, classifier refitted & %.2f & %.2f\\\\\n'%(ref_ov,ref_69)
t5+='CNN predicts accumulated turning, threshold & -- & %.2f\\\\\n'%scal_69
t5+='\\midrule\nDescriptor from the recorded trajectory & %.2f & %.2f\\\\\n'%(ours_ov,ours_69)
t5+='\\botrule\n\\end{tabular}\n\\end{table}\n'
w('t5_offline.tex',t5)

t6=r"""\begin{table}[t]
\caption{Robustness of the proposed descriptor on the Pendigits test set.
Position noise is Gaussian with the stated standard deviation on the coordinate
scale of $0$ to $100$. Dropped positions are removed from the interior of the
path, which is then resampled to eight points. Values are per cent.}
\label{tab:robust}
\begin{tabular}{lcc}
\toprule
Condition & all digits & digits 6 and 9\\
\midrule
"""
t6+='No perturbation & %.2f & %.2f\\\\\n'%tuple(j4['clean'])
for s in ('0.5','1','2','4'):
    if s in j4['jitter']: t6+='Position noise, sd $=%s$ & %.2f & %.2f\\\\\n'%(s,*j4['jitter'][s])
for n in ('1','2'):
    if n in j4['dropped']: t6+='%s interior position(s) dropped & %.2f & %.2f\\\\\n'%(n,*j4['dropped'][n])
for f,lab in [('0.25','a quarter'),('0.5','half'),('1.0','all')]:
    if f in j4['wrong_order']: t6+='Stroke order reversed for %s of the test set & %.2f & %.2f\\\\\n'%(lab,*j4['wrong_order'][f])
t6+='Mirrored input, signed descriptor & %.2f & %.2f\\\\\n'%tuple(j4['mirror_ours'])
t6+='Mirrored input, unsigned variant & %.2f & %.2f\\\\\n'%tuple(j4['mirror_nosign'])
t6+='\\botrule\n\\end{tabular}\n\\end{table}\n'
w('t6_robust.tex',t6)

# ---------- macros ----------
m=open(f'{OUT}/tables/macros.tex','w')
def mac(k,v): m.write('\\newcommand{\\%s}{%s}\n'%(k,v))
mac('ourover','%.2f'%ours_ov); mac('oursixnine','%.2f'%ours_69)
mac('cnnaugsixnine','%.1f'%cnn_69); mac('cnnaugoverall','%.1f'%cnn_ov)
mac('riccsixnine','%.1f'%imean('RIC-CNN (upright train)',1).mean())
mac('zernsixnine','%.1f'%z_69.mean()); mac('zernoverall','%.2f'%z_ov.mean())
mac('rawupright','%.2f'%raw_ov[0]); mac('rawworst','%.2f'%raw_ov.min()); mac('rawsixnineup','%.2f'%raw_69[0])
mac('nfeat',str(fin['ours']['nfeat'])); mac('maxdev','$1.4\\times10^{-14}$')
mac('nosignsixnine','%.2f'%np.mean(ns['acc69'])); mac('noordersixnine','%.2f'%np.mean(no['acc69']))
mac('pnoorder','7.1\\times10^{-15}'); mac('pnosign','%.2f'%fin['mc_nosign']['p'])
pr=np.array(o['pred0']); Kp=np.isin(yte,[6,9])
mac('npair',str(int(Kp.sum()))); mac('nsixnineerr',str(int(((pr!=yte)&Kp).sum())))
mac('nswap',str(int((((yte==6)&(pr==9))|((yte==9)&(pr==6))).sum())))
mac('totalturnsix','%.2f'%sc['mean6']); mac('totalturnnine','%.2f'%sc['mean9'])
mac('sdturnsix','%.2f'%sc['sd6']); mac('sdturnnine','%.2f'%sc['sd9'])
mac('scalarrule','%.2f'%sc['test_acc']); mac('scalarthr','%.2f'%sc['thr'])
# MNIST control
mac('mncnnaugoverall','%.2f'%bo.mean()); mac('mncnnaugsixnine','%.1f'%b6.mean())
mac('mncnnauglowoverall','%.2f'%ao.mean()); mac('mncnnauglowsixnine','%.1f'%a6.mean())
mac('mnstatoverall','%.2f'%mn_st[0].mean()); mac('mnstatsixnine','%.1f'%mn_st[1].mean())
mac('mncnnuprightsixnine','%.1f'%np.mean(mn_up[1]))
# offline recovery
mac('offlineoverall','%.2f'%off_ov); mac('offlinesixnine','%.2f'%off_69)
mac('offlinerefitoverall','%.2f'%ref_ov); mac('offlinerefitsixnine','%.2f'%ref_69)
mac('offlinescalarsixnine','%.2f'%scal_69)
mac('scalarpredsix','%.2f'%d6); mac('scalarprednine','%.2f'%d9)
# sequence models
mac('seqdev','$%.0e$'%j2['invariance_dev'] if j2['invariance_dev']>0 else '$0$')
mac('seqcnn','%.2f'%np.mean(j2['cnn1d']['acc'])); mac('seqcnnsixnine','%.2f'%np.mean(j2['cnn1d']['acc69']))
mac('seqlstm','%.2f'%np.mean(j2['bilstm']['acc'])); mac('seqlstmsixnine','%.2f'%np.mean(j2['bilstm']['acc69']))
# robustness and cost
mac('jitterone','%.2f'%j4['jitter']['1'][1]); mac('wrongorderall','%.2f'%j4['wrong_order']['1.0'][1])
mac('mirrorours','%.2f'%j4['mirror_ours'][1]); mac('mirrornosign','%.2f'%j4['mirror_nosign'][1])
mac('costdesc','%.0f'%j4['cost']['desc_us_per_sample']); mac('costfit','%.1f'%j4['cost']['fit_s'])
mac('costpred','%.0f'%j4['cost']['predict_us_per_sample'])
# skeleton rule on MNIST
mac('sciupright','%.1f'%j5['upright']); mac('scirot','%.1f'%j5['rotated']); mac('sciflip','%.0f'%j5['flip'])
mac('mnpair',str(int(j5['n'])))
m.close()
print(open(f'{OUT}/tables/macros.tex').read())
print('tables and figures written')
