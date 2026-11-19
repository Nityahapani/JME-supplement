"""Exact (rational-arithmetic) check of Lemma 5.2 for specific parameters.
For sigma rational and prices chosen so that all fractional powers are rational, the N_f functions of Lemma 5.2 are
rational functions of the free coordinates (r, omega). Their exact rank over Q at sample points equals N_f iff the
functions are linearly independent *for these parameters* (rank N_f at some sample points proves independence).
Runtime: a few seconds."""
import random, math
from fractions import Fraction as Fr
from sympy.polys.matrices import DomainMatrix
from sympy import QQ

def sqrtfr(x):                       # exact square root of a perfect-square Fraction
    a,b=x.numerator,x.denominator; ra,rb=math.isqrt(a),math.isqrt(b); assert ra*ra==a and rb*rb==b; return Fr(ra,rb)
def powfr(x,s):                      # x**s for s in {1/2, 3/2, 1} and x a perfect square (or any x if s==1)
    if s==Fr(1): return x
    if s==Fr(1,2): return sqrtfr(x)
    if s==Fr(3,2): return sqrtfr(x)*x
    raise ValueError

def features(p,pp,sigma,r,om):
    n=len(p); L=n+1
    def block(pr):
        pt=list(pr)+[Fr(1)]
        w=sum(om[m]*pt[m] for m in range(L))
        xi=[powfr(pr[k],1-sigma) if 1-sigma>0 else 1/powfr(pr[k],sigma-1) if sigma!=1 else Fr(1) for k in range(n)]
        D=1+sum(xi[k]*r[k] for k in range(n))
        cj=[1/powfr(pr[j],sigma) for j in range(n)]                   # p_j^{-sigma}
        cj1=[cj[j]/pr[j] for j in range(n)]                           # p_j^{-sigma-1}
        F=[w*r[j]*cj[j]/D-om[j] for j in range(n)]
        dF=[[ (om[k]*cj[j]-sigma*(1 if j==k else 0)*w*cj1[j])*r[j]/D-(1-sigma)*w*cj[j]*cj[k]*r[j]*r[k]/D**2 for k in range(n)] for j in range(n)]
        return F,dF
    F,dF=block(p); Fp,dFp=block(pp)
    row=[Fr(1)]+F+Fp+[dF[j][k] for j in range(n) for k in range(n)]+[dFp[j][k] for j in range(n) for k in range(n)]+[om[j] for j in range(n)]
    return row

def rank_at(L,sigma,p,pp,seed=1):
    random.seed(seed); n=L-1; Nf=1+2*n+2*n*n+n; rows=[]
    for _ in range(3*Nf):
        r=[Fr(random.randint(1,9),random.randint(1,9)) for _ in range(n)]
        raw=[Fr(random.randint(1,9),1) for _ in range(L)]; s=sum(raw); om=[x/s for x in raw]
        rows.append(features(p,pp,sigma,r,om))
    M=DomainMatrix([[QQ(x.numerator,x.denominator) for x in row] for row in rows],(len(rows),Nf),QQ)
    return Nf,M.rank()

sq={2:([Fr(4)],[Fr(9)]),3:([Fr(4),Fr(9)],[Fr(16),Fr(25)]),4:([Fr(4),Fr(9),Fr(25)],[Fr(16),Fr(36),Fr(49)])}
print("Exact rank over Q of the N_f functions of Lemma 5.2 (full rank <=> independent for these parameters)")
print(" L  sigma   p, p'                          N_f  exact rank")
for L in (2,3,4):
    p,pp=sq[L]
    for sigma in (Fr(1,2),Fr(3,2),Fr(1)):
        Nf,rk=rank_at(L,sigma,p,pp)
        tag="  <-- sigma=1: rank collapses (hyperplanes coincide)" if sigma==1 else ("" if rk==Nf else "  <-- DEFICIENT")
        print(f" {L}  {str(sigma):5s}  {[str(x) for x in p]},{[str(x) for x in pp]}".ljust(48)+f"{Nf:4d}  {rk:4d}{tag}")
