import numpy as np, time
import iproj
from iproj import Ifun, zcons
from scipy.special import logsumexp
def build_trunc(N,sig0,sighet,kappa,tau,smin,smax,seed):
    rng=np.random.default_rng(seed); L=3
    la=tau*rng.standard_normal((N,L)); la-=logsumexp(la,axis=1,keepdims=True)
    om=rng.dirichlet(kappa*np.ones(L),size=N); lom=np.log(np.clip(om,1e-14,None))
    sig=np.clip(sig0*np.exp(sighet*rng.standard_normal(N)),smin,smax)
    return la,lom,om,sig
if __name__=="__main__":
    import sys
    smin=float(sys.argv[1]); sig0=float(sys.argv[2])
    T=build_trunc(1500,sig0,0.7,0.2,0.5,smin,3.0,2026)
    t=time.time()
    print(f"sigma truncated to [{smin},3.0], sig0={sig0}, sighet=0.7: min I over angle/phi/h at radius r")
    for Rr in [2,4,6,8,12,16,24,36,60,100]:
        best=(9,None)
        for ang in np.linspace(0,2*np.pi,16,endpoint=False):
            q=Rr*np.array([np.cos(ang),np.sin(ang)])
            for ph in np.linspace(0,np.pi,6,endpoint=False):
                for h in [0.1,1.0]:
                    with np.errstate(all='ignore'):
                        I=Ifun(q,ph,h,T)
                    if np.isfinite(I) and I<best[0]: best=(I,(round(ang,2),round(ph,2),h))
        print(f"r={Rr:4d} min I={best[0]:.4f} at {best[1]}  t={time.time()-t:.0f}s",flush=True)
