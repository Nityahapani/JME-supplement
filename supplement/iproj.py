import numpy as np, time, sys
from scipy.special import logsumexp
from scipy.optimize import minimize

def build(N, sig0, sighet, kappa, tau, seed):
    rng=np.random.default_rng(seed); L=3
    la=tau*rng.standard_normal((N,L)); la-=logsumexp(la,axis=1,keepdims=True)
    om=rng.dirichlet(kappa*np.ones(L),size=N); lom=np.log(np.clip(om,1e-14,None))
    sig=sig0*np.exp(sighet*rng.standard_normal(N))
    return la,lom,om,sig

def zcons(q,T):
    la,lom,om,sig=T; qq=np.append(q,0.0)
    lw=logsumexp(lom+qq,axis=1)
    lden=logsumexp(sig[:,None]*la+(1-sig[:,None])*qq,axis=1)
    lx=lw[:,None]+sig[:,None]*(la-qq)-lden[:,None]
    return np.exp(lx)-om                      # (N,3) per-consumer excess demand

def Ifun(q1,phi,h,T,return_all=False):
    d=np.array([np.cos(phi),np.sin(phi)])
    z1=zcons(q1,T)[:,:2]
    if h<1e-6: raise ValueError
    z2=zcons(q1+h*d,T)[:,:2]
    f=np.hstack([z1,(z2-z1)/h])
    sd=f.std(0)+1e-12; f=f/sd
    N=len(f)
    def g(l):
        a=f@l; m=logsumexp(a); w=np.exp(a-m)
        return m-np.log(N), f.T@w
    best=None
    r=minimize(g,np.zeros(4),jac=True,method='L-BFGS-B',bounds=[(-80,80)]*4,options={'maxiter':200})
    I=-r.fun
    if return_all:
        a=f@r.x; w=np.exp(a-logsumexp(a)); return I,w,r.x,sd
    return I

if __name__=="__main__":
    N=1500; sig0,sighet,kappa,tau=0.2,1.0,0.2,0.5
    T=build(N,sig0,sighet,kappa,tau,seed=2026)
    t=time.time(); R=12.0
    gq=np.arange(-R,R+0.1,1.5); phis=np.linspace(0,np.pi,6,endpoint=False); hs=[0.05,0.5,2.0]
    res=[]
    for x in gq:
        for y in gq:
            for ph in phis:
                for h in hs:
                    res.append((Ifun(np.array([x,y]),ph,h,T),x,y,ph,h))
        if time.time()-t>200: print("time cut at x=",x); break
    res.sort()
    print("grid evals",len(res),"time",round(time.time()-t))
    print("best coarse:",[tuple(np.round(r,3)) for r in res[:6]])
    np.save("coarse.npy",np.array(res[:30]))
