import numpy as np, time, itertools, sys
from l3eng import Grid
def cyc_types(sig,e,d,dp):
    ty=[]
    for i in range(3):
        om=np.full(3,e); om[i]=1-2*e
        a=np.full(3,0.0); a[i]=d; a[(i+1)%3]=1-d-dp; a[(i+2)%3]=dp
        ty.append((a,sig,om))
    return ty
if __name__=="__main__":
    t0=time.time(); hits=[]
    print("cyclic economy scan: N equilibria at centroid nu=(1/3,1/3,1/3)  [sigma, eps(off-endow), d(own taste), d'(prev taste)]")
    for sig in [0.1,0.15,0.2,0.3,0.5]:
        for e in [0.002,0.02,0.1]:
            for d,dp in [(0.02,0.02),(0.1,0.1),(0.02,0.2)]:
                R=min(60,12/sig+8); G=Grid(cyc_types(sig,e,d,dp),R=R,G=241)
                eq=G.equilibria(np.ones(3)/3); n=len(eq); s=sum(x[1] for x in eq)
                if n>=3: hits.append((sig,e,d,dp,n,s)); print(f"  HIT sigma={sig} e={e} d={d} dp={dp}: N={n} index_sum={s} ({time.time()-t0:.0f}s)",flush=True)
        if time.time()-t0>250: break
    print("scan done; hits:",len(hits),f"({time.time()-t0:.0f}s)")
