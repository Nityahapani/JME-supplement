import numpy as np, time
from scipy.stats import binom
from scipy.optimize import minimize
from l2mix import economy_K, sample_cluster, A0, B0
import l2
from ip5 import inner
noise=0.03
# ---------- exact-ish conditional MC for P(M) ----------
rng=np.random.default_rng(10); t=time.time(); qtab={}
for M in [20,40,80,160,320]:
    for K in range(int(0.36*M),int(0.45*M)+1):
        k=0; n=100
        for _ in range(n):
            a,s,w=economy_K(rng,M,K,noise); k+= l2.count(a,s,w)>=3
        qtab[(M,K)]=k/n
    if time.time()-t>110: break
def P(M,pi):
    ks=[K for (m,K) in qtab if m==M]; return sum(binom.pmf(K,M,pi)*qtab[(M,K)] for K in ks)
print(f"MC done ({time.time()-t:.0f}s). band indicator q(K,M) nonzero for K/M in:",
      {M:(min([K/M for (m,K),v in qtab.items() if m==M and v>0.5],default=None),max([K/M for (m,K),v in qtab.items() if m==M and v>0.5],default=None)) for M in [20,40,80,160,320]})
# ---------- I-projection (L=2, scalar excess demand) ----------
def atoms(pi,N,seed):
    r=np.random.default_rng(seed); K=r.binomial(N,pi)
    aA,sA,wA=sample_cluster(r,K,A0,noise); aB,sB,wB=sample_cluster(r,N-K,B0,noise)
    return np.r_[aA,aB],np.r_[sA,sB],np.r_[wA,wB]
def zc(q,T):
    a,s,w=T; p=np.exp(q); inc=w*p+(1-w)
    x=inc*a**s*p**(-s)/(a**s*p**(1-s)+(1-a)**s)
    return x-w, x+w
def Ival(q1,h,T):
    z1,g1=zc(q1,T); z2,g2=zc(q1+h,T)
    f=np.stack([z1,(z2-z1)/h],1); I,wt,ok=inner(f)
    rel=np.max(np.abs(wt@f)/(wt@np.abs(f)+1e-300)) if ok else 9
    rel2=max(abs(wt@z1)/(wt@g1),abs(wt@z2)/(wt@g2)) if ok else 9
    return I,wt,ok,rel2
def Iproj(pi,N=6000,seed=3):
    T=atoms(pi,N,seed); best=(np.inf,None)
    def obj(x):
        q1=x[0]; h=float(np.clip(np.exp(x[1]),5e-3,10))
        if abs(q1)>35: return 60
        with np.errstate(all='ignore'): I,w,ok,rel=Ival(q1,h,T)
        return min(I,60) if ok and np.isfinite(I) and rel<1e-7 else 60
    for q1 in np.arange(-30,31,1.0):
        for h in [0.01,0.1,1.0]:
            v=obj(np.array([q1,np.log(h)]))
            if v<best[0]: best=(v,np.array([q1,np.log(h)]))
    r=minimize(obj,best[1],method='Nelder-Mead',options={'xatol':1e-6,'fatol':1e-9,'maxiter':300})
    q1=r.x[0]; h=float(np.clip(np.exp(r.x[1]),5e-3,10)); I,w,ok,rel=Ival(q1,h,T)
    return I,q1,h,rel
KL=lambda a,b: a*np.log(a/b)+(1-a)*np.log((1-a)/(1-b))
for pi,edge in [(0.20,0.3955),(0.65,0.4151)]:
    I,q1,h,rel=Iproj(pi)
    print(f"\nmu: share of type A = {pi}.  I-projection rate = {I:.4f}  (q1={q1:.2f}, sep={h:.3f}, rel.resid={rel:.1e});  noise-free Cramer rate = {KL(edge,pi):.4f}")
    for M in [20,40,80,160,320]:
        if not any(m==M for (m,K) in qtab): continue
        p=P(M,pi); print(f"   M={M:4d}  P(multiple)={p:.3e}   -logP/M={-np.log(p)/M if p>0 else np.inf:.4f}")
    ms=[M for M in [20,40,80,160,320] if any(m==M for (m,K) in qtab) and P(M,pi)>0]
    if len(ms)>=2:
        a,b=ms[-2],ms[-1]; sl=-(np.log(P(b,pi))-np.log(P(a,pi)))/(b-a)
        print(f"   slope of -logP between M={a} and M={b}: {sl:.4f}")
print("time",round(time.time()-t))
