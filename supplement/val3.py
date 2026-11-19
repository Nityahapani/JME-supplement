import numpy as np, time
from sim3 import find_grid
from sim2 import jac
from iproj3 import build_trunc
from iproj import zcons
def strict_roots(E,r,qmax=25.0):
    out=[]
    for q,idx,st in r:
        z=zcons(q,E); rel=np.max(np.abs(z.sum(0)[:2])/(np.abs(z).sum(0)[:2]+1e-300))
        J=jac(q,E); sv=np.linalg.svd(J,compute_uv=False); cond=sv[-1]/sv[0]
        if rel<1e-8 and cond>1e-6 and np.max(np.abs(q))<=qmax: out.append(idx)
    return out
t=time.time(); n=200
print("clean estimate, truncated ensemble L=3 (strict roots: rel.resid<1e-8, cond>1e-6, |q|<=25); 200 economies each")
print("M | P(certain multiple) [Wilson95] | economies with no strict root | pred. exp(-1.93 M)")
for M in [2,3,4,6,8,12]:
    k=0; none=0
    for i in range(n):
        E=build_trunc(M,0.2,1.0,0.2,0.5,0.08,3.0,seed=900000+1000*M+i)
        r=find_grid(E,ng=161,topk=60); ids=strict_roots(E,r)
        if len(ids)==0: none+=1
        k+= int(len(ids)>=2 or (len(ids)>=1 and sum(ids)!=1))
    p=k/n; z=1.96; c=(p+z*z/(2*n))/(1+z*z/n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    print(f"{M:2d} | {p:.3f} [{max(0,c-h):.3f},{c+h:.3f}] | {none}/{n} | {np.exp(-1.93*M):.1e}   ({time.time()-t:.0f}s)",flush=True)
    if time.time()-t>250: break
