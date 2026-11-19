import numpy as np, time
from scipy.optimize import minimize, root
from ip5 import Ival
from iproj import zcons
from iproj3 import build_trunc
T=build_trunc(1500,0.2,1.0,0.2,0.5,0.08,3.0,2026); N=1500
Rb=8.0
def obj(x):
    q=x[:2]; ph=x[2]; h=float(np.clip(np.exp(x[3]),1e-2,8.0))
    if np.abs(q).max()>Rb or np.abs(q+h*np.array([np.cos(ph),np.sin(ph)])).max()>Rb: return 60.0
    with np.errstate(all='ignore'): I,w,ok,f=Ival(q,ph,h,T)
    if not (ok and np.isfinite(I)): return 60.0
    rel=np.max(np.abs(w@f)/(w@np.abs(f)+1e-300)); return min(I,60.0) if rel<1e-7 else 60.0
rng=np.random.default_rng(7); pts=[]
for _ in range(500):
    x=np.array([rng.uniform(-Rb,Rb),rng.uniform(-Rb,Rb),rng.uniform(0,np.pi),np.log(10**rng.uniform(-1,0.7))]); pts.append((obj(x),x))
pts.sort(key=lambda p:p[0]); best=None
for v,x0 in pts[:4]:
    r=minimize(obj,x0,method='Nelder-Mead',options={'maxiter':250,'xatol':1e-6,'fatol':1e-9})
    if best is None or r.fun<best.fun: best=r
x=best.x; q1=x[:2]; ph=x[2]; h=float(np.clip(np.exp(x[3]),1e-2,8.0)); q2=q1+h*np.array([np.cos(ph),np.sin(ph)])
I,w,ok,f=Ival(q1,ph,h,T)
Fq=lambda q:(w[:,None]*zcons(q,T)[:,:2]).sum(0)
Fs=lambda q:(w[:,None]*np.abs(zcons(q,T)[:,:2])).sum(0)
def jacrel(q,hh=1e-6):
    J=np.zeros((2,2)); f0=Fq(q)/Fs(q)
    for k in range(2):
        dq=q.copy(); dq[k]+=hh; J[:,k]=(Fq(dq)/Fs(dq)-f0)/hh
    return J
roots=[]; rr=np.random.default_rng(5)
for _ in range(600):
    q0=rr.uniform(-Rb-2,Rb+2,2)
    rt=root(lambda q:Fq(q)/Fs(q),q0,method='hybr',tol=1e-14)
    if rt.success and np.abs(Fq(rt.x)/Fs(rt.x)).max()<1e-11 and np.abs(rt.x).max()<Rb+4 and all(np.abs(rt.x-r_[0]).max()>1e-4 for r_ in roots):
        J=jacrel(rt.x); sv=np.linalg.svd(J,compute_uv=False); roots.append((rt.x,int(np.sign(np.linalg.det(J))),sv[-1]/sv[0],np.abs(Fq(rt.x)/Fs(rt.x)).max()))
print(f"minimizer of the L=3 I-projection (no floor, price box {Rb}): I={I:.4f}; designed zeros at q1={np.round(q1,3)}, q2={np.round(q2,3)} (separation {h:.3f})")
print(f"all equilibria of the reweighted economy nu* found by 600 multistarts in the box (relative tolerance):")
for r_ in roots: print(f"   q={np.round(r_[0],3)}  index(sgn det J)={r_[1]:+d}  cond-ratio sv_min/sv_max={r_[2]:.2e}  rel.resid={r_[3]:.1e}")
print("sum of indices:",sum(r_[1] for r_ in roots))
Fm=lambda q:(zcons(q,T)[:,:2]).mean(0); Fsm=lambda q:np.abs(zcons(q,T)[:,:2]).mean(0)
sm=[]
for _ in range(600):
    q0=rr.uniform(-Rb-2,Rb+2,2)
    rt=root(lambda q:Fm(q)/Fsm(q),q0,method='hybr',tol=1e-14)
    if rt.success and np.abs(Fm(rt.x)/Fsm(rt.x)).max()<1e-11 and np.abs(rt.x).max()<Rb+4 and all(np.abs(rt.x-s)<1e-4 for s in [] ) or (rt.success and np.abs(Fm(rt.x)/Fsm(rt.x)).max()<1e-11 and all(np.abs(rt.x-s).max()>1e-4 for s in sm) and np.abs(rt.x).max()<Rb+4): sm.append(rt.x)
print("equilibria of the original population mu in the same box:",len(sm),[np.round(s,3) for s in sm])
