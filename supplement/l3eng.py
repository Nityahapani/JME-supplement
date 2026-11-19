import numpy as np, time
from scipy.special import logsumexp
from scipy.optimize import root

def type_z(q, a, sig, om):
    """per-consumer excess demand and gross scale at log-prices q (...,2) with q3=0; returns z (...,3), gross (...,3)"""
    qq=np.concatenate([q,np.zeros(q.shape[:-1]+(1,))],-1)
    lom=np.log(np.clip(om,1e-14,None)); la=np.log(a)
    lw=logsumexp(lom+qq,axis=-1)
    lden=logsumexp(sig*la+(1-sig)*qq,axis=-1)
    lx=lw[...,None]+sig*(la-qq)-lden[...,None]
    x=np.exp(lx); return x-om, x+om

class Grid:
    def __init__(self,types,R=20.0,G=401):
        self.types=types; self.R=R; self.G=G
        g=np.linspace(-R,R,G); Q1,Q2=np.meshgrid(g,g,indexing='ij')
        self.Q=np.stack([Q1,Q2],-1)
        self.Z=[];self.S=[]
        for (a,s,om) in types:
            z,gr=type_z(self.Q,a,s,om); self.Z.append(z[...,:2]); self.S.append(gr[...,:2])
    def rel(self,nu):
        Z=sum(n*z for n,z in zip(nu,self.Z)); S=sum(n*s for n,s in zip(nu,self.S)); return Z/S
    def point_rel(self,q,nu):
        tot=np.zeros(2); sc=np.zeros(2)
        for n,(a,s,om) in zip(nu,self.types):
            z,gr=type_z(np.asarray(q),a,s,om); tot+=n*z[:2]; sc+=n*gr[:2]
        return tot/sc
    def equilibria(self,nu,tol=1e-9,screen=False):
        r=self.rel(nu); thr=1e-12
        sg=np.where(np.abs(r)<thr,0,np.sign(r))
        c=[sg[:-1,:-1],sg[1:,:-1],sg[:-1,1:],sg[1:,1:]]
        mx=np.max(c,axis=0); mn=np.min(c,axis=0); mixed=(mx>0)&(mn<0)
        cand=mixed[...,0]&mixed[...,1]
        from scipy import ndimage
        lab,nl=ndimage.label(cand)
        if screen and nl<2: return []
        idx=np.array([np.argwhere(lab==k)[0] for k in range(1,min(nl,40)+1)]) if nl>0 else np.zeros((0,2),int); roots=[]
        g=np.linspace(-self.R,self.R,self.G)
        for i,j in idx[:400]:
            q0=np.array([0.5*(g[i]+g[i+1]),0.5*(g[j]+g[j+1])])
            rt=root(lambda q:self.point_rel(q,nu),q0,method='hybr',tol=1e-13)
            q=rt.x
            if not rt.success or np.abs(self.point_rel(q,nu)).max()>tol or np.abs(q).max()>self.R: continue
            if any(np.abs(q-s[0]).max()<2e-3 for s in roots): continue
            h=1e-6; J=np.zeros((2,2)); f0=self.point_rel(q,nu)
            for k in range(2):
                dq=q.copy(); dq[k]+=h; J[:,k]=(self.point_rel(dq,nu)-f0)/h
            sv=np.linalg.svd(J,compute_uv=False)
            if sv[-1]/sv[0]<1e-7: continue
            roots.append((q,int(np.sign(np.linalg.det(J)))))
        return roots

def rand_types(rng,K=3):
    types=[]
    for _ in range(K):
        a=rng.dirichlet([1.2,1.2,1.2]); a=np.clip(a,0.03,None); a/=a.sum()
        om=rng.dirichlet([0.25,0.25,0.25]); om=np.clip(om,1e-3,None); om/=om.sum()
        sig=10**rng.uniform(np.log10(0.12),np.log10(0.6))
        types.append((a,sig,om))
    return types

if __name__=="__main__":
    rng=np.random.default_rng(7); t0=time.time(); found=[]
    nus=[]; m=6
    for i in range(1,m):
        for j in range(1,m-i): nus.append(np.array([i,j,m-i-j])/m)
    ntry=0
    while time.time()-t0<240:
        ntry+=1; types=rand_types(rng); G=Grid(types,R=20,G=201)
        best=0; bnu=None
        for nu in nus:
            eq=G.equilibria(nu,screen=True); n=len(eq); s=sum(e[1] for e in eq)
            if n>=3 and s==1 and n>best: best=n; bnu=nu
        if best>=3: found.append((types,bnu,best)); print(f"hit after {ntry} tries: N={best} at nu={np.round(bnu,3)} ({time.time()-t0:.0f}s)",flush=True)
        if len(found)>=3: break
    print("tries",ntry,"found",len(found))
    import pickle; pickle.dump(found,open("l3found.pkl","wb"))
