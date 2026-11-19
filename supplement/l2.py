import numpy as np, time
Q=np.linspace(-80,80,6001); P1=np.exp(Q)
def sample_types(rng,n,sig0,kap=0.3,gam=0.7,tau=0.5,sh=0.5,smin=0.08,smax=3.0):
    w1=np.clip(rng.beta(kap,kap,n),1e-6,1-1e-6)
    lg=np.log(w1/(1-w1))
    a=1/(1+np.exp(-(-gam*lg+tau*rng.standard_normal(n))))
    a=np.clip(a,1e-4,1-1e-4)
    s=np.clip(sig0*np.exp(sh*rng.standard_normal(n)),smin,smax)
    return a,s,w1
def Zmean(a,s,w1,wt=None):
    inc=w1[:,None]*P1[None,:]+(1-w1[:,None])
    num=a[:,None]**s[:,None]*P1[None,:]**(-s[:,None])
    den=a[:,None]**s[:,None]*P1[None,:]**(1-s[:,None])+(1-a[:,None])**s[:,None]
    z=inc*num/den-w1[:,None]
    scale=inc*num/den+w1[:,None]
    if wt is None: wt=np.ones(len(a))/len(a)
    return (wt[:,None]*z).sum(0), (wt[:,None]*scale).sum(0)
def count(a,s,w1):
    Z,S=Zmean(a,s,w1); r=Z/S
    r=r[np.abs(r)>1e-9]
    return int((np.diff(np.sign(r))!=0).sum())
if __name__=="__main__":
    rng=np.random.default_rng(1); t=time.time()
    print("L=2 ensemble tuning: P(N>=3) by M and sigma0 (4000 economies each)")
    for sig0 in [0.08,0.12,0.2]:
        row=[]
        for M in [2,4,8,16]:
            k=0; n=2000
            for _ in range(n):
                a,s,w1=sample_types(rng,M,sig0); k+= count(a,s,w1)>=3
            row.append(k/n)
        print(f"sig0={sig0}: ",["%.3f"%x for x in row],f"({time.time()-t:.0f}s)",flush=True)
