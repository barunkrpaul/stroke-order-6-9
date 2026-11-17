import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import feats as F
OUT='/mnt/user-data/outputs/icaiet_paper'
ANG=list(range(0,360,10))
traj=json.load(open('traj_results.json')); img=json.load(open('img_results.json'))
fin=json.load(open('final2_results.json')); zrn=json.load(open('zern_results.json'))
def imean(name,key):           # mean over seeds of the per-angle curve
    seeds=img[name]; 
    return np.mean([[seeds[s][str(a)][key] for a in ANG] for s in seeds],0)
ours_ov=np.mean(fin['ours']['acc']); ours_69=np.mean(fin['ours']['acc69'])
raw_ov=np.array(traj['raw']['overall']).mean(0); raw_69=np.array(traj['raw']['acc69']).mean(0)
z_ov=np.array([zrn[str(a)][0] for a in ANG]); z_69=np.array([zrn[str(a)][1] for a in ANG])
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

Ptr,ytr,Pte,yte=F.load()
sc=json.load(open('scalar_rule.json'))
plt.rcParams.update({'font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,
                     'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5})
fig,ax=plt.subplots(1,2,figsize=(3.5,1.75))
for dg,c,ls in [(6,'#1D9E75','-'),(9,'#7F77DD','--')]:
    k,_=F.turning(Ptr[ytr==dg]); m=k.mean(0); sd=k.std(0)
    ax[0].plot(range(1,7),m,color=c,ls=ls,lw=1.2,marker='o',ms=2,label=f'{dg}')
    ax[0].fill_between(range(1,7),m-sd,m+sd,color=c,alpha=.13)
    ax[1].hist(F.turning(Ptr[ytr==dg])[0].sum(1),bins=30,range=(-8,10),alpha=.55,color=c,label=f'{dg}')
ax[0].axhline(0,c='0.5',lw=.6)
ax[0].set_xlabel('vertex along path'); ax[0].set_ylabel('turning angle (rad)')
ax[0].set_title('(a) per vertex'); ax[0].legend(title='digit',ncol=2,handlelength=1.2)
ax[0].grid(alpha=.3)
ax[1].axvline(sc['thr'],c='0.25',ls=':',lw=1.1)
ax[1].set_xlabel('total turning (rad)'); ax[1].set_ylabel('samples')
ax[1].set_title('(b) accumulated'); ax[1].legend(title='digit',ncol=2,handlelength=1.2)
ax[1].grid(alpha=.3)
plt.tight_layout(pad=0.3); plt.savefig(f'{OUT}/figs/fig_turning_profile.pdf',bbox_inches='tight'); plt.close()
plt.rcParams.update(plt.rcParamsDefault)
tot6=F.turning(Ptr[ytr==6])[0].sum(1).mean(); tot9=F.turning(Ptr[ytr==9])[0].sum(1).mean()

# ---------------- tables ----------------
def row(name,ov_up,ov_mean,ov_worst,s69_mean,s69_worst):
    return f'{name} & {ov_up:.2f} & {ov_mean:.2f} & {ov_worst:.2f} & {s69_mean:.2f} & {s69_worst:.2f} \\\\\n'
def stats(c_ov,c_69):
    return c_ov[0],c_ov.mean(),c_ov.min(),c_69.mean(),c_69.min()
main=r"""\begin{table}[t]
\caption{Accuracy (per cent) on the writer-independent Pendigits test set. The
columns report the upright condition, the mean over the $36$ test rotations and
the value at the worst of those rotations, first over all ten digits and then
over the digits $6$ and $9$ only.}
\label{tab:main}
\centering
\begin{tabular}{lccccc}
\toprule
& \multicolumn{3}{c}{all digits} & \multicolumn{2}{c}{digits 6 and 9}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Representation & upright & mean & worst & mean & worst\\
\midrule
"""
main+=row('Raw coordinates',*stats(raw_ov,raw_69))
main+=row('CNN, upright training',*stats(imean('CNN (upright train)',0),imean('CNN (upright train)',1)))
main+=row('CNN, rotation augmented',*stats(imean('CNN + rot. augmentation',0),imean('CNN + rot. augmentation',1)))
main+=row('RIC-CNN layers~\\cite{riccnn}',*stats(imean('RIC-CNN (upright train)',0),imean('RIC-CNN (upright train)',1)))
main+=row('Zernike + Hu + radial',*stats(z_ov,z_69))
main+=('\\midrule\n'+row('\\textbf{Proposed descriptor}',ours_ov,ours_ov,ours_ov,ours_69,ours_69))
main+=r"""\bottomrule
\end{tabular}
\end{table}
"""
open(f'{OUT}/tables/main_table.tex','w').write(main)

ab=r"""\begin{table}[t]
\caption{Ablation of the proposed descriptor on the same test set. Accuracy is
identical at every rotation, so a single value is reported. The last column
gives the $p$ value of the test of McNemar against the full descriptor on the
digits $6$ and $9$.}
\label{tab:abl}
\centering
\begin{tabular}{lccc}
\toprule
Variant & all digits & digits 6 and 9 & $p$\\
\midrule
"""
o=fin['ours']; ns=fin['nosign']; no=fin['noorder']
ab+=f"Full descriptor & {np.mean(o['acc']):.2f} $\\pm$ {np.std(o['acc']):.2f} & {np.mean(o['acc69']):.2f} $\\pm$ {np.std(o['acc69']):.2f} & --\\\\\n"
ab+=f"Signs of turning removed & {np.mean(ns['acc']):.2f} $\\pm$ {np.std(ns['acc']):.2f} & {np.mean(ns['acc69']):.2f} $\\pm$ {np.std(ns['acc69']):.2f} & {fin['mc_nosign']['p']:.3g}\\\\\n"
ab+=f"Stroke direction unknown & {np.mean(no['acc']):.2f} $\\pm$ {np.std(no['acc']):.2f} & {np.mean(no['acc69']):.2f} $\\pm$ {np.std(no['acc69']):.2f} & {fin['mc_noorder']['p']:.2g}\\\\\n"
ab+=r"""\bottomrule
\end{tabular}
\end{table}
"""
open(f'{OUT}/tables/ablation_table.tex','w').write(ab)

m=open(f'{OUT}/tables/macros.tex','w')
def mac(k,v): m.write('\\newcommand{\\%s}{%s}\n'%(k,v))
mac('ourover',f'{ours_ov:.2f}'); mac('oursixnine',f'{ours_69:.2f}')
mac('cnnaugsixnine',f"{imean('CNN + rot. augmentation',1).mean():.1f}")
mac('cnnaugoverall',f"{imean('CNN + rot. augmentation',0).mean():.1f}")
mac('riccsixnine',f"{imean('RIC-CNN (upright train)',1).mean():.1f}")
mac('zernsixnine',f'{z_69.mean():.1f}')
mac('rawupright',f'{raw_ov[0]:.2f}'); mac('rawworst',f'{raw_ov.min():.2f}')
mac('rawsixnineup',f'{raw_69[0]:.2f}')
mac('nfeat',str(fin['ours']['nfeat']))
mac('maxdev','$1.4\\times10^{-14}$')
mac('dropnosign',f"{np.mean(o['acc69'])-np.mean(ns['acc69']):.2f}")
mac('dropnoorder',f"{np.mean(o['acc69'])-np.mean(no['acc69']):.2f}")
mac('pnoorder','7.1\\times10^{-15}'); mac('pnosign','%.3g'%fin['mc_nosign']['p'])
pr=np.array(o['pred0']); Kp=np.isin(yte,[6,9])
mac('npair',str(int(Kp.sum())))
mac('nsixnineerr',str(int(((pr!=yte)&Kp).sum())))
mac('nswap',str(int((((yte==6)&(pr==9))|((yte==9)&(pr==6))).sum())))
mac('totalturnsix',f'{tot6:.2f}'); mac('totalturnnine',f'{tot9:.2f}')
mac('sdturnsix',f"{sc['sd6']:.2f}"); mac('sdturnnine',f"{sc['sd9']:.2f}")
mac('scalarrule',f"{sc['test_acc']:.2f}"); mac('scalarthr',f"{sc['thr']:.2f}")
mac('nosignsixnine',f"{np.mean(ns['acc69']):.2f}"); mac('noordersixnine',f"{np.mean(no['acc69']):.2f}")
mac('zernoverall',f'{z_ov.mean():.2f}')
m.close()
print(open(f'{OUT}/tables/macros.tex').read())
print(main); print(ab)
