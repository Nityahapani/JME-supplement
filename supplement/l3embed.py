import numpy as np, time, itertools
from l3eng import Grid
def embed(d3,e3,sigC,tasteC=(0.25,0.25,0.5),eC=0.001):
    A=(np.array([0.95*(1-d3),0.05*(1-d3),d3]),0.08,np.array([0.999*(1-e3),0.001*(1-e3),e3]))
    B=(np.array([0.05*(1-d3),0.95*(1-d3),d3]),0.5, np.array([0.001*(1-e3),0.999*(1-e3),e3]))
    # A: a=0.95 on good1, sigma .08, owns good1 ; B: a=0.05 on good1 (prefers good 2), sigma .5, owns good2
    C=(np.array(tasteC),sigC,np.array([eC,eC,1-2*eC]))
    return [A,B,C]
if __name__=="__main__":
    t0=time.time()
    print("L=3 embedding of the L=2 multiplicity economy: N(equilibria) over (share of C, A-share among A,B)")
    for d3,e3,sigC in [(0.01,0.001,0.5),(0.05,0.01,0.5),(0.02,0.002,0.2),(0.02,0.002,1.0)]:
        G=Grid(embed(d3,e3,sigC),R=60,G=481)
        print(f"d3={d3} e3={e3} sigC={sigC}")
        for nC in [0.005,0.02,0.05,0.1,0.2]:
            row=[]
            for rA in [0.25,0.28,0.30,0.32,0.34,0.36,0.40]:
                nu=np.array([rA*(1-nC),(1-rA)*(1-nC),nC]); eq=G.equilibria(nu); row.append(f"{rA:.2f}:{len(eq)}")
            print("  nC=%.3f "%nC," ".join(row),f"({time.time()-t0:.0f}s)",flush=True)
        if time.time()-t0>240: break
