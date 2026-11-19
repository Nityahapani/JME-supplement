import numpy as np, time, sys
from scipy.optimize import minimize, root
from scipy.special import logsumexp
from iproj import zcons
from iproj3 import build_trunc

def inner_floor(f,g,maxit=150):
    N=len(f); X=np.hstack([f,g]); d=X.shape[1]; nf=f.shape[1]
    x=np.zeros(d); idx=np.arange(d)
    G=lambda v: logsumexp(X@v)-np.log(N)
    for it in range(maxit):
        a=X@x; m=logsumexp(a); w=np.exp(a-m); grad=X.T@w
        H=(X*w[:,None]).T@X-np.outer(grad,grad)
        atb=(idx>=nf)&(x<=1e-13)&(grad>0)
        fr=~atb
        step=np.zeros(d)
        try: step[fr]=np.linalg.solve(H[np.ix_(fr,fr)]+1e-13*np.trace(H)*np.eye(fr.sum()),grad[fr])
        except np.linalg.LinAlgError: return np.inf,w,False
        dec=grad[fr]@step[fr]
        if m-np.log(N)<-25: return np.inf,w,False
        if dec<1e-13: break
        t=1.0; g0=m-np.log(N)
        while t>1e-14:
            xn=x-t*step; xn[nf:]=np.maximum(xn[nf:],0.0)
            if G(xn)<=g0-0.25*grad@(x-xn): break
            t*=0.5
        if t<=1e-14: break
        x=xn
    a=X@x; m=logsumexp(a); w=np.exp(a-m)
    return -(m-np.log(N)),w,True

def Ival(q1,ph,h,T,c):
    d=np.array([np.cos(ph),np.sin(ph)])
    z1=zcons(q1,T)[:,:2]; z2=zcons(q1+h*d,T)[:,:2]
    f=np.hstack([z1,(z2-z1)/h]); om=T[2]
    I,w,ok=inner_floor(f,om-c)
    return I,w,ok,f

def check(w,f,om,c):
    rel=np.max(np.abs(w@f)/(w@np.abs(f)+1e-300))
    return rel, (w@om).min()

def run(T,c,Rb,ns,ntop,seed,tmax):
    def obj(x):
        q=x[:2]; ph=x[2]; h=float(np.clip(np.exp(x[3]),1e-2,8.0))
        if np.abs(q).max()>Rb or np.abs(q+h*np.array([np.cos(ph),np.sin(ph)])).max()>Rb: return 60.0
        with np.errstate(all='ignore'): I,w,ok,f=Ival(q,ph,h,T,c)
        if not (ok and np.isfinite(I)): return 60.0
        rel,fl=check(w,f,T[2],c)
        if rel>1e-7 or fl<c-1e-7: return 60.0
        return min(I,60.0)
    rng=np.random.default_rng(seed); t=time.time(); pts=[]
    for _ in range(ns):
        x=np.array([rng.uniform(-Rb,Rb),rng.uniform(-Rb,Rb),rng.uniform(0,np.pi),np.log(10**rng.uniform(-1,0.7))])
        pts.append((obj(x),x))
        if time.time()-t>tmax*0.4: break
    pts.sort(key=lambda p:p[0]); best=None
    for v,x0 in pts[:ntop]:
        r=minimize(obj,x0,method='Nelder-Mead',options={'maxiter':200,'xatol':1e-5,'fatol':1e-8})
        if best is None or r.fun<best.fun: best=r
        if time.time()-t>tmax: break
    x=best.x; q1=x[:2]; ph=x[2]; h=float(np.clip(np.exp(x[3]),1e-2,8.0)); q2=q1+h*np.array([np.cos(ph),np.sin(ph)])
    I,w,ok,f=Ival(q1,ph,h,T,c); rel,fl=check(w,f,T[2],c)
    N=len(w); KL=float(np.sum(w*np.log(np.clip(w*N,1e-300,None))))
    return dict(I=I,KL=KL,q1=q1,q2=q2,h=h,rel=rel,floor=fl,w=w)

if __name__=="__main__":
    T=build_trunc(1500,0.2,1.0,0.2,0.5,0.08,3.0,2026)
    cs=[float(a) for a in sys.argv[1].split(',')]; Rbs=[float(a) for a in sys.argv[2].split(',')]
    print("c, box | I_c  primalKL | rel.resid  min agg endow | q1 q2 sep",flush=True)
    for c in cs:
        for Rb in Rbs:
            t=time.time(); r=run(T,c,Rb,300,3,int(100*c+Rb),55)
            print(f"{c:.2f} {Rb:4.0f} | {r['I']:.4f} {r['KL']:.4f} | {r['rel']:.1e} {r['floor']:.3f} | {np.round(r['q1'],2)} {np.round(r['q2'],2)} {r['h']:.2f} ({time.time()-t:.0f}s)",flush=True)
