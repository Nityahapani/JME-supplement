"""Two-type, two-good exact example (Section 6.1 of the paper).
Reproduces: multiplicity band, high-precision tangency edges, exact binomial decay tables, critical case.
Runtime: about 1-2 minutes."""
import numpy as np, mpmath as mp
from scipy.optimize import brentq
from scipy.stats import binom
A=(0.756,0.082,0.991); B=(0.063,0.291,0.005)   # (taste weight on good 1, elasticity sigma, endowment share of good 1)
def zx(q,a,s,w1):
    # excess demand and gross scale (x+w1) of good 1 at log relative price q = log(p1/p2), CES demand
    p=np.exp(q); inc=w1*p+(1-w1); x=inc*a**s*p**(-s)/(a**s*p**(1-s)+(1-a)**s); return x-w1, x+w1
def rel(q,nu):                                   # aggregate excess demand normalised by gross demand + endowment
    zA,gA=zx(q,*A); zB,gB=zx(q,*B); return (nu*zA+(1-nu)*zB)/(nu*gA+(1-nu)*gB)
def count(nu,q):
    r=rel(q,nu); r=r[np.abs(r)>1e-12]; return int((np.diff(np.sign(r))!=0).sum())
for (R,Gp) in [(60,24001),(60,240001),(150,240001)]:
    q=np.linspace(-R,R,Gp); nus=np.linspace(0.005,0.995,991); N=np.array([count(n,q) for n in nus]); idx=np.where(N>=3)[0]
    print(f"grid R={R}, {Gp} pts: max N={N.max()}, N>=3 on nu in [{nus[idx[0]]:.3f},{nus[idx[-1]]:.3f}], single-type economies N={N[0]},{N[-1]}")
qg=np.linspace(-60,60,240001)
def edge(x_in,x_out):
    for _ in range(50):
        m=0.5*(x_in+x_out); x_in,x_out=(m,x_out) if count(m,qg)>=3 else (x_in,m)
    return x_in
lo0=edge(0.40,0.30); hi0=edge(0.40,0.50); print("grid-bisection band edges:",lo0,hi0)
def roots_at(nu):
    r=rel(qg,nu); m=np.abs(r)>1e-12; qq=qg[m]; rr=r[m]; idx=np.where(np.diff(np.sign(rr))!=0)[0]
    return [brentq(lambda t:rel(t,nu),qq[i],qq[i+1],xtol=1e-14) for i in idx]
mp.mp.dps=40
def Fmp(qq,nu):
    def z(a,s,w1):
        p=mp.e**qq; inc=w1*p+(1-w1); return inc*mp.mpf(a)**s*p**(-s)/(mp.mpf(a)**s*p**(1-s)+(1-mp.mpf(a))**s)-w1
    return nu*z(mp.mpf(A[0]),mp.mpf(A[1]),mp.mpf(A[2]))+(1-nu)*z(mp.mpf(B[0]),mp.mpf(B[1]),mp.mpf(B[2]))
edges=[]
for nu_in in [lo0+2e-5,hi0-2e-5]:
    rts=sorted(roots_at(nu_in)); k=int(np.argmin(np.diff(rts))); q0=0.5*(rts[k]+rts[k+1])   # the close pair is the nearly merged double root
    sol=mp.findroot(lambda qq,nn:(Fmp(qq,nn),mp.diff(lambda t:Fmp(t,nn),qq)),(mp.mpf(q0),mp.mpf(nu_in)),tol=1e-25,verify=False)
    edges.append((float(sol[1]),float(sol[0])))
lo,hi=edges[0][0],edges[1][0]
print(f"tangency solve (40 digits): band = [{lo:.6f},{hi:.6f}]; double-root log-prices {edges[0][1]:.4f}, {edges[1][1]:.4f}")
KL=lambda a,b:a*np.log(a/b)+(1-a)*np.log((1-a)/(1-b))
for mu in [0.20,0.65]:
    e=lo if mu<lo else hi; print(f"mu_A={mu}: Cramer rate {KL(e,mu):.5f}")
    for M in [100,1000,3000]:
        K=np.arange(M+1); ins=(K/M>=lo)&(K/M<=hi); P=binom.pmf(K,M,mu)[ins].sum(); print(f"   M={M}: P={P:.3e}  -logP/M={-np.log(P)/M:.5f}")
print("critical case and just-outside case:")
for name,mu in [("mu=lo",lo),("mu=hi",hi),("mu=lo-0.01",lo-0.01)]:
    row=[]
    for M in [100,1000,10000,100000]:
        K=np.arange(M+1); ins=(K/M>=lo)&(K/M<=hi); row.append(binom.pmf(K,M,mu)[ins].sum())
    print("  ",name,["%.3e"%x for x in row])
print("inside the band (mu=0.405):",[ (M,round(float(binom.pmf(np.arange(M+1),M,0.405)[(np.arange(M+1)/M>=lo)&(np.arange(M+1)/M<=hi)].sum()),3)) for M in [10,30,100,300,1000,3000,10000]])
