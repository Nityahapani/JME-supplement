import numpy as np, sys, time, warnings
from scipy.optimize import root
from scipy.special import logsumexp
from sim2 import make_economy, Zred, jac
warnings.filterwarnings("ignore")

def Zgrid(Q1,Q2,E):
    la,lom,om,sig=E
    Q=np.stack([Q1,Q2,np.zeros_like(Q1)],-1)                    # (...,3)
    lw=logsumexp(lom[None,None]+Q[:,:,None,:],axis=-1)          # (n,n,M)
    lden=logsumexp(sig[:,None]*la+(1-sig[:,None])*Q[:,:,None,:],axis=-1)
    lx=lw[...,None]+sig[:,None]*(la-Q[:,:,None,:])-lden[...,None]
    Z=np.exp(lx).sum(2)-om.sum(0)
    return Z[...,:2]

def find_grid(E, ng=161, topk=60):
    sig=E[3]; styp=np.exp(np.mean(np.log(sig)))
    R=min(8/styp+8,60)
    g=np.linspace(-R,R,ng); Q1,Q2=np.meshgrid(g,g,indexing='ij')
    Z=Zgrid(Q1,Q2,E); nrm=np.sqrt((Z**2).sum(-1))
    # local minima of norm
    P=np.pad(nrm,1,constant_values=np.inf); loc=np.ones_like(nrm,bool)
    for di in(-1,0,1):
        for dj in(-1,0,1):
            if di==0 and dj==0: continue
            loc&=nrm<=P[1+di:1+di+ng,1+dj:1+dj+ng]
    pts=np.argwhere(loc); vals=nrm[loc]; order=np.argsort(vals)[:topk]
    roots=[]
    for k in order:
        q0=np.array([g[pts[k][0]],g[pts[k][1]]])
        r=root(Zred,q0,args=(E,),method='hybr',tol=1e-13)
        if not r.success: continue
        q=r.x
        if np.max(np.abs(Zred(q,E)))>1e-9 or np.max(np.abs(q))>R*1.5: continue
        if all(np.max(np.abs(q-s[0]))>1e-5*(1+np.max(np.abs(q))) for s in roots):
            J=jac(q,E); idx=int(np.sign(np.linalg.det(-J)))
            st=bool(np.all(np.linalg.eigvals(J).real<0)); roots.append((q,idx,st))
    return roots

def run(M,sig0,sighet,kappa,tau,necon,seed=1):
    rng=np.random.default_rng(seed); out=[]
    for _ in range(necon):
        E=make_economy(rng,3,M,sig0,sighet,kappa,tau); r=find_grid(E)
        out.append((len(r),sum(x[1] for x in r)==1,sum(x[2] for x in r)))
    return np.array(out)

if __name__=="__main__":
    necon=int(sys.argv[1]); M=int(sys.argv[2]); kappa=float(sys.argv[3]); tau=float(sys.argv[4])
    t=time.time(); print(f"L=3 M={M} necon={necon} kappa={kappa} tau={tau}")
    print("sig0 sighet | complete | among complete: P(N=1) P(N>=3) mean N max N meanStable")
    for sighet in [0.0,1.0]:
        for sig0 in [0.1,0.2,0.4,0.8]:
            o=run(M,sig0,sighet,kappa,tau,necon); ok=o[:,1]==1; c=o[ok,0]
            print(f"{sig0:4.2f} {sighet:4.1f} | {ok.mean():.2f} | {np.mean(c==1):.3f} {np.mean(c>=3):.3f} {c.mean():.2f} {c.max()} {o[ok,2].mean():.2f}",flush=True)
    print("time",round(time.time()-t))
