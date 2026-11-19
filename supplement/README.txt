SUPPLEMENT: code for "How likely is multiplicity? A large-deviation rate ..."
Python 3.12; numpy, scipy, sympy, mpmath. Scripts import each other, so run them from this folder.
Scripts print their results; I did not save logs except toy_two_type.log.

Paper section -> script(s)
 6.1 two-type, two-good exact example          toy_two_type.py  (log: toy_two_type.log)
                                               toy2x2d.py       (structured search that found the economy; random seed 5)
 6.2 continuous types, two goods (approx.)     l2.py (counter), l2mix.py (ensemble), l2val.py (conditional Monte Carlo + I-projection;
                                               uses ip5.py inner Newton solver). Runtime about 4 min.
 6.3 three goods
     certified upper bounds on the rate        ip7.py (price-box sensitivity, c=0), ip9.py (floors; argument = comma-separated floors, then box)
                                               e.g.  python3 ip9.py 0.05,0.10,0.15,0.20 20 ;  ip10.py (stronger search)
     optimal reweighting has 3 equilibria      g3check.py
     direct simulation, strict filtering       val3.py  (needs sim2.py, sim3.py)
     negative results (engineered economies)   l3eng.py, l3cyc.py, l3embed.py
 Ensemble builders: iproj.py, iproj3.py (build_trunc = the clipped-elasticity L=3 ensemble, seed 2026).

Conventions: the inner solver ip5.inner is a damped Newton method on the dual; a result is accepted only if it converged,
the dual and primal values agree, and the relative constraint residual is below 1e-7 (checked in ip7/ip9/ip10/g3check).
Equilibrium counts are accepted only with (i) residual normalised by gross demand + endowment, (ii) Jacobian condition
ratio above 1e-6, (iii) a price box, (iv) index sum = +1 where applicable.

 Exact-arithmetic check of Lemma 5.2 (L=2,3,4)  exact_independence.py  (a few seconds)
