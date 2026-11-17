"""Step-by-step visualisation of the method, for the paper."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import feats as F
from scipy.interpolate import CubicSpline
Ptr,ytr,Pte,yte=F.load()

def smooth(P,n=200):
    d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))]; t=d/d[-1]
    u,ix=np.unique(t,return_index=True)
    return CubicSpline(u,P[ix],axis=0)(np.linspace(0,1,n))
P6,P9=Ptr[699],Ptr[6884]          # samples closest to the class centre in descriptor space

def turn(P):
    v=np.diff(P,axis=0); n=np.linalg.norm(v,axis=1,keepdims=True); u=v/np.where(n<1e-9,1e-9,n)
    cr=u[:-1,0]*u[1:,1]-u[:-1,1]*u[1:,0]; dt=(u[:-1]*u[1:]).sum(1)
    return np.arctan2(cr,dt),u

plt.rcParams.update({'font.size':7.2,'axes.titlesize':7.6,'axes.labelsize':7.2,
                     'xtick.labelsize':6.4,'ytick.labelsize':6.4,'legend.fontsize':6.4})
fig,ax=plt.subplots(1,4,figsize=(7.4,2.0))

# (a) ordered pen positions
for P,c,lab,mk in [(P6,'#1D9E75','digit 6','o'),(P9,'#7F77DD','digit 9','s')]:
    S=smooth(P); ax[0].plot(S[:,0],S[:,1],'-',color=c,lw=1.6,alpha=.55)
    ax[0].scatter(P[:,0],P[:,1],s=13,color=c,marker=mk,zorder=3,label=lab)
    ax[0].scatter(*P[0],s=42,facecolor='none',edgecolor=c,lw=1.0,zorder=4)
    for k,(x,y) in enumerate(P):
        ax[0].annotate(str(k),(x,y),textcoords='offset points',xytext=(3,2.5),fontsize=5.4,color=c)
ax[0].set_title('(a) pen positions in writing order')
ax[0].legend(loc='lower center',bbox_to_anchor=(0.5,-0.22),ncol=2,handlelength=1.0,frameon=False)
ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].set_aspect('equal')
ax[0].text(0.02,0.02,'ring: first position',transform=ax[0].transAxes,fontsize=5.4,color='0.35')

# (b) unit tangents and one signed turning angle
k6,u6=turn(P6)
S6=smooth(P6); ax[1].plot(S6[:,0],S6[:,1],'-',color='0.82',lw=1.6,zorder=1)
ax[1].plot(P6[:,0],P6[:,1],'-',color='0.62',lw=0.7,zorder=1)
sc=0.13*(P6[:,0].max()-P6[:,0].min()+1e-9)
for j in range(len(u6)):
    ax[1].annotate('',xy=(P6[j,0]+u6[j,0]*sc,P6[j,1]+u6[j,1]*sc),xytext=(P6[j,0],P6[j,1]),
                   arrowprops=dict(arrowstyle='-|>',color='#1D9E75',lw=.9,shrinkA=0,shrinkB=0))
jm=int(np.argmax(np.abs(k6)))
ax[1].scatter(*P6[jm+1],s=44,facecolor='none',edgecolor='#C8734B',lw=1.2,zorder=4)
ax[1].annotate(r'$\kappa_{%d}=%+.2f$ rad'%(jm,k6[jm]),P6[jm+1],textcoords='offset points',
               xytext=(7,-11),fontsize=6.0,color='#C8734B')
ax[1].set_title('(b) unit tangents and turning')
ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].set_aspect('equal')

# (c) cumulative turning, original and rotated
for P,c,lab,ls in [(P6,'#1D9E75','digit 6','-'),(P9,'#7F77DD','digit 9','--')]:
    kr,_=turn(F.rotate(P[None],137)[0])
    ax[2].plot(range(1,len(kr)+1),np.cumsum(kr),'-',color=c,lw=4.0,alpha=.25,solid_capstyle='round')
    k,_=turn(P)
    ax[2].plot(range(1,len(k)+1),np.cumsum(k),ls,color=c,lw=1.3,marker='o',ms=2.6,label=lab)
ax[2].axhline(0,c='0.6',lw=.6)
ax[2].set_xlabel('vertex index'); ax[2].set_ylabel('cumulative turning (rad)')
ax[2].set_title('(c) cumulative turning'); ax[2].legend(handlelength=1.3,loc='upper left')
ax[2].grid(alpha=.3)
ax[2].text(0.03,0.03,'thick band: the same paths rotated by $137^{\\circ}$',
           transform=ax[2].transAxes,fontsize=5.5,color='0.35')

# (d) descriptor composition
blocks=[('turning',30,'#1D9E75'),('lengths',18,'#7F77DD'),
        ('distances',28,'#EF9F27'),('areas',6,'#C8734B')]
x=0
for nm,w,c in blocks:
    ax[3].barh(0,w,left=x,height=.42,color=c,alpha=.9,edgecolor='white')
    ax[3].text(x+w/2,0,str(w),ha='center',va='center',fontsize=6.0,color='white')
    ax[3].text(x+w/2,0.30,nm.replace('\n',' '),ha='center',va='bottom',fontsize=5.2,color=c,rotation=0)
    x+=w
ax[3].set_xlim(0,x); ax[3].set_ylim(-.55,.55); ax[3].set_yticks([])
ax[3].set_xlabel('descriptor component index')
ax[3].set_title('(d) descriptor, %d components'%x)
ax[3].set_ylim(-.55,.85)
plt.tight_layout(pad=0.4)
for out in ('/mnt/user-data/outputs/icaiet_paper/figs/fig_method.pdf',
            '/mnt/user-data/outputs/paaa_journal/figs/fig_method.pdf'):
    plt.savefig(out,bbox_inches='tight')
plt.close()
k6t=np.cumsum(turn(P6)[0])[-1]; k9t=np.cumsum(turn(P9)[0])[-1]
print('method figure written. total turning of the two shown samples: 6 -> %.2f, 9 -> %.2f'%(k6t,k9t))
