import numpy as np, time
from ip9 import run
from iproj3 import build_trunc
T=build_trunc(1500,0.2,1.0,0.2,0.5,0.08,3.0,2026)
for c in [0.0,0.10]:
    t=time.time(); r=run(T,c,20.0,2000,10,7,125)
    print(f"c={c:.2f} box 20 (stronger search): I={r['I']:.4f} primalKL={r['KL']:.4f} rel.resid={r['rel']:.1e} min agg endow={r['floor']:.3f} q1={np.round(r['q1'],2)} q2={np.round(r['q2'],2)} sep={r['h']:.2f} ({time.time()-t:.0f}s)",flush=True)
