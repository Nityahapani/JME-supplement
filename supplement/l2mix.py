import numpy as np, time
import l2
l2.Q=np.linspace(-40,40,2401); l2.P1=np.exp(l2.Q)
from l2 import count
A0=(0.756,0.082,0.991); B0=(0.063,0.291,0.005)
lg=lambda x: np.log(x/(1-x)); sg=lambda x: 1/(1+np.exp(-x))
def sample_cluster(rng,n,p0,noise):
    a,s,w=p0
    return (np.clip(sg(lg(a)+noise*rng.standard_normal(n)),1e-4,1-1e-4),
            np.clip(s*np.exp(noise*rng.standard_normal(n)),0.05,3.0),
            np.clip(sg(lg(w)+2*noise*rng.standard_normal(n)),1e-6,1-1e-6))
def economy_K(rng,M,K,noise):
    aA,sA,wA=sample_cluster(rng,K,A0,noise); aB,sB,wB=sample_cluster(rng,M-K,B0,noise)
    return np.r_[aA,aB],np.r_[sA,sB],np.r_[wA,wB]
if __name__=="__main__":
    rng=np.random.default_rng(2); t=time.time()
    for noise in [0.0,0.03,0.08]:
        print(f"noise={noise}: P(N>=3 | share of A) at M=60 (80 draws each)")
        row=[]
        for nu in np.arange(0.30,0.52,0.02):
            K=int(round(nu*60)); k=0
            for _ in range(80):
                a,s,w=economy_K(rng,60,K,noise); k+= count(a,s,w)>=3
            row.append(f"{K/60:.2f}:{k/80:.2f}")
        print(" ",row,f"({time.time()-t:.0f}s)",flush=True)
