import numpy as np
from scipy.stats import binom
rng=np.random.default_rng(5)
q=np.linspace(-150,150,12001); p1=np.exp(q)
def z1(a,s,w1):
    inc=w1*p1+(1-w1)
    x1=inc*a**s*p1**(-s)/(a**s*p1**(1-s)+(1-a)**s)
    return x1-w1
def count(nu,A,B,ZA=None,ZB=None):
    ZA=z1(*A) if ZA is None else ZA; ZB=z1(*B) if ZB is None else ZB
    Z=nu*ZA+(1-nu)*ZB
    sg=np.sign(Z[np.abs(Z)>1e-9]); return int((np.diff(sg)!=0).sum())
found=[]
for t in range(30000):
    A=(rng.uniform(.02,.98),rng.uniform(.08,.35),rng.uniform(.8,.999))
    B=(rng.uniform(.02,.98),rng.uniform(.08,.35),rng.uniform(.001,.2))
    ZA,ZB=z1(*A),z1(*B)
    if count(0,A,B,ZA,ZB)!=1 or count(1,A,B,ZA,ZB)!=1: continue
    c=[count(nu,A,B,ZA,ZB) for nu in np.linspace(.1,.9,9)]
    if max(c)>=3:
        found.append((A,B,c))
        if len(found)>=40: break
print("economies with an interior multiplicity found:",len(found),"of draws",t+1)
if not found: raise SystemExit
nus=np.linspace(0.002,0.998,300)
best=None
for A,B,c in found:
    ZA,ZB=z1(*A),z1(*B)
    N=np.array([count(nu,A,B,ZA,ZB) for nu in nus]); idx=np.where(N>=3)[0]
    if len(idx)==0: continue
    # require a single contiguous block
    w=nus[idx[-1]]-nus[idx[0]]
    if best is None or w>best[0]: best=(w,A,B,N)
w,A,B,N=best; ZA,ZB=z1(*A),z1(*B)
idx=np.where(N>=3)[0]; lo,hi=nus[idx[0]],nus[idx[-1]]
def edge(x_in,x_out):
    for _ in range(40):
        m=(x_in+x_out)/2
        if count(m,A,B,ZA,ZB)>=3: x_in=m
        else: x_out=m
    return x_in
lo=edge(lo,lo-0.0034); hi=edge(hi,hi+0.0034)
print("type A (taste on good1, elasticity, endowment share good1):",np.round(A,3))
print("type B:",np.round(B,3))
print(f"N>=3 on share-of-A in [{lo:.4f},{hi:.4f}]; max N={N.max()}; contiguous={ (N[idx[0]:idx[-1]+1]>=3).all() }")
print("single-type economies: N =",count(0,A,B,ZA,ZB),count(1,A,B,ZA,ZB))
def KL(a,b): return a*np.log(a/b)+(1-a)*np.log((1-a)/(1-b))
mids=[0.5*lo, 0.5*(1+hi)]
for mu in mids+[0.5*(lo+hi)]:
    inside=lo<=mu<=hi; nstar=lo if mu<lo else (hi if mu>hi else mu)
    I=0 if inside else KL(nstar,mu)
    print(f"\nmu_A={mu:.3f} inside={inside}  Cramer rate I={I:.4f}")
    for M in [10,30,100,300,1000,3000,10000]:
        K=np.arange(M+1); ins=(K/M>=lo)&(K/M<=hi)
        P=binom.pmf(K,M,mu)[ins].sum()
        print(f"  M={M:6d} P(multiple)={P:.3e} -logP/M={(-np.log(P)/M) if P>0 else np.inf:.4f}")
