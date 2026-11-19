import numpy as np, sys, time, warnings
from scipy.optimize import root
from scipy.special import logsumexp
warnings.filterwarnings("ignore")

def make_economy(rng, L, M, sig0, sighet, kappa, tau):
    la = tau*rng.standard_normal((M,L)); la -= logsumexp(la,axis=1,keepdims=True)
    om = rng.dirichlet(kappa*np.ones(L), size=M)
    lom = np.log(np.clip(om,1e-14,None))
    sig = sig0*np.exp(sighet*rng.standard_normal(M))
    return la, lom, om, sig

def Zfull(q, E):
    la,lom,om,sig=E
    qq=np.append(q,0.0)
    lw = logsumexp(lom+qq,axis=1)                                   # (M,)
    lden = logsumexp(sig[:,None]*la + (1-sig[:,None])*qq, axis=1)   # (M,)
    lx = lw[:,None] + sig[:,None]*(la-qq) - lden[:,None]
    return np.exp(lx).sum(0) - om.sum(0)

def Zred(q,E): return Zfull(q,E)[:-1]

def jac(q,E,h=1e-6):
    n=len(q); J=np.zeros((n,n)); f0=Zred(q,E)
    for k in range(n):
        dq=q.copy(); dq[k]+=h; J[:,k]=(Zred(dq,E)-f0)/h
    return J

def find(E, L, rng, nstart):
    sig=E[3]; styp=np.exp(np.mean(np.log(sig)))
    s0=min(1.5/styp,25); cut=8/styp+8
    roots=[]
    for _ in range(nstart):
        q0=s0*rng.standard_normal(L-1)
        r=root(Zred,q0,args=(E,),method='hybr',tol=1e-13)
        if not r.success: continue
        q=r.x
        if np.max(np.abs(Zred(q,E)))>1e-9 or np.max(np.abs(q))>cut: continue
        if all(np.max(np.abs(q-s[0]))>1e-5*(1+np.max(np.abs(q))) for s in roots):
            J=jac(q,E); idx=int(np.sign(np.linalg.det(-J)))
            st=bool(np.all(np.linalg.eigvals(J).real<0))
            roots.append((q,idx,st))
    return roots

def run(L,M,sig0,sighet,kappa,tau,necon,nstart,seed=1):
    rng=np.random.default_rng(seed); out=[]
    for _ in range(necon):
        E=make_economy(rng,L,M,sig0,sighet,kappa,tau)
        r=find(E,L,rng,nstart)
        out.append((len(r), sum(x[1] for x in r)==1, sum(x[2] for x in r), sum(1 for x in r if x[1]==1), sum(1 for x in r if x[1]==-1)))
    return np.array(out)

if __name__=="__main__":
    L=int(sys.argv[1]); necon=int(sys.argv[2]); nstart=int(sys.argv[3]); M=int(sys.argv[4]) if len(sys.argv)>4 else 2*L
    t=time.time(); print(f"L={L} M={M} necon={necon} nstart={nstart}")
    print("sig0 sighet kappa tau | complete | among complete: P(N=1) P(N>=3) mean N max N meanStable")
    for sighet in [0.0,1.0]:
        for sig0 in [0.1,0.2,0.4,0.8]:
            for kappa in [0.2]:
                o=run(L,M,sig0,sighet,kappa,0.5,necon,nstart)
                ok=o[:,1]==1; c=o[ok,0]
                if len(c)==0: print(sig0,sighet,"no complete"); continue
                print(f"{sig0:4.2f} {sighet:4.1f} {kappa:4.1f} 0.5 | {ok.mean():.2f} | {np.mean(c==1):.3f} {np.mean(c>=3):.3f} {c.mean():.2f} {c.max()} {o[ok,2].mean():.2f}",flush=True)
    print("time",round(time.time()-t))
