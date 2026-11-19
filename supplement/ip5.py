import numpy as np, time
from scipy.special import logsumexp
from iproj import zcons
from iproj3 import build_trunc

def inner(f, tol=1e-9, maxit=80):
    """min_lambda log mean exp(f@lambda); returns I=-min, weights, converged flag. Newton + backtracking."""
    N,d=f.shape; lam=np.zeros(d)
    def g(l):
        a=f@l; return logsumexp(a)-np.log(N)
    gv=0.0
    for it in range(maxit):
        a=f@lam; m=logsumexp(a); w=np.exp(a-m)
        grad=f.T@w
        mu=grad; C=(f*w[:,None]).T@f-np.outer(mu,mu)
        # affine-invariant convergence: Newton decrement
        try: step=np.linalg.solve(C+1e-14*np.trace(C)*np.eye(d),grad)
        except np.linalg.LinAlgError: return np.inf,w,False
        dec=grad@step
        if dec<tol: return -(m-np.log(N)),w,True
        t=1.0; g0=m-np.log(N)
        while t>1e-12 and g(lam-t*step)>g0-0.25*t*dec: t*=0.5
        lam=lam-t*step
        if m-np.log(N)<-25: return np.inf,w,False   # diverging: 0 outside hull -> I=inf
    return -(g(lam)),w,False

def Ival(q1,phi,h,T):
    d=np.array([np.cos(phi),np.sin(phi)])
    z1=zcons(q1,T)[:,:2]; z2=zcons(q1+h*d,T)[:,:2]
    f=np.hstack([z1,(z2-z1)/h])
    f=f-0.0
    I,w,ok=inner(f)
    return I,w,ok,f

if __name__=="__main__":
    T=build_trunc(1500,0.6,0.7,0.2,0.5,0.4,3.0,2026)
    t=time.time(); print("radius profile with converged Newton solver (inf = infeasible)")
    for Rr in [2,4,6,8,12,16,24,36,60]:
        best=(np.inf,None,0,0)
        for ang in np.linspace(0,2*np.pi,16,endpoint=False):
            q=Rr*np.array([np.cos(ang),np.sin(ang)])
            for ph in np.linspace(0,np.pi,6,endpoint=False):
                for h in [0.1,1.0]:
                    with np.errstate(all='ignore'): I,w,ok,f=Ival(q,ph,h,T)
                    if ok and I<best[0]: best=(I,(round(ang,2),round(ph,2),h),0,0)
        print(f"r={Rr:3d} min I={best[0]:.4f} at {best[1]}  t={time.time()-t:.0f}s",flush=True)
        if time.time()-t>240: break
