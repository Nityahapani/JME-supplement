import numpy as np, time
from scipy.optimize import minimize, root
from ip5 import Ival
from iproj import zcons
from iproj3 import build_trunc
T=build_trunc(1500,0.2,1.0,0.2,0.5,0.08,3.0,2026)
def make_obj(Rb):
    def obj(x):
        q=x[:2]; ph=x[2]; h=float(np.clip(np.exp(x[3]),1e-2,8.0))
        if np.abs(q).max()>Rb or np.abs(q+h*np.array([np.cos(ph),np.sin(ph)])).max()>Rb: return 60.0
        with np.errstate(all='ignore'): I,w,ok,f=Ival(q,ph,h,T)
        return min(I,60.0) if ok and np.isfinite(I) else 60.0
    return obj
def opt(Rb,ns,ntop,seed,tmax):
    obj=make_obj(Rb); rng=np.random.default_rng(seed); t=time.time(); pts=[]
    for _ in range(ns):
        x=np.array([rng.uniform(-Rb,Rb),rng.uniform(-Rb,Rb),rng.uniform(0,np.pi),np.log(10**rng.uniform(-1,0.7))])
        pts.append((obj(x),x))
        if time.time()-t>tmax*0.4: break
    pts.sort(key=lambda p:p[0]); best=None
    for v,x0 in pts[:ntop]:
        r=minimize(obj,x0,method='Nelder-Mead',options={'maxiter':250,'xatol':1e-5,'fatol':1e-8})
        if best is None or r.fun<best.fun: best=r
        if time.time()-t>tmax: break
    return best
for Rb in [8.0,40.0]:
    t=time.time(); b=opt(Rb,500,4,int(Rb),100)
    q1=b.x[:2]; h=float(np.clip(np.exp(b.x[3]),1e-2,8.0)); ph=b.x[2]; q2=q1+h*np.array([np.cos(ph),np.sin(ph)])
    print(f"box |q|<={Rb:4.0f}: I*={b.fun:.4f}  q1={np.round(q1,2)} q2={np.round(q2,2)}  ({time.time()-t:.0f}s)",flush=True)
# verification of the Rb=20 solution with relative scaling
x=np.load("ip6_best.npy"); q1=x[:2]; ph=x[2]; h=float(np.clip(np.exp(x[3]),1e-2,8.0)); q2=q1+h*np.array([np.cos(ph),np.sin(ph)])
I,w,ok,f=Ival(q1,ph,h,T)
def Zrel(q):
    z=zcons(q,T); return (w[:,None]*z).sum(0)[:2]/((w[:,None]*np.abs(z)).sum(0)[:2]+1e-300)
print("Rb=20 solution: I=%.4f"%I,"relative residuals at q1,q2:",np.abs(Zrel(q1)).max(),np.abs(Zrel(q2)).max())
for name,q in [("q1",q1),("q2",q2)]:
    with np.errstate(all='ignore'): r=root(Zrel,q,method='hybr',tol=1e-14)
    print(name,"polished root",np.round(r.x,4),"moved by",np.abs(r.x-q).max(),"rel resid",np.abs(Zrel(r.x)).max())
Zm=lambda q:(zcons(q,T).mean(0)[:2]/((np.abs(zcons(q,T)).mean(0)[:2])+1e-300))
S=[]
r0=np.random.default_rng(2)
for _ in range(300):
    with np.errstate(all='ignore'): r=root(Zm,r0.normal(0,10,2),method='hybr',tol=1e-14)
    if r.success and np.abs(Zm(r.x)).max()<1e-8 and all(np.abs(r.x-s).max()>1e-3 for s in S): S.append(r.x)
print("equilibria of mu itself (relative tolerance):",len(S),[np.round(s,2) for s in S])
