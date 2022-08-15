# ShiftedProximalOperators

[![CI](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl/actions/workflows/ci.yml)
[![](https://img.shields.io/badge/docs-latest-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/ShiftedProximalOperators.jl/dev)
[![codecov](https://codecov.io/gh/JuliaSmoothOptimizers/ShiftedProximalOperators.jl/branch/master/graph/badge.svg?token=CZzi6ufcXI)](https://codecov.io/gh/JuliaSmoothOptimizers/ShiftedProximalOperators.jl)
[![DOI](https://zenodo.org/badge/357330060.svg)](https://zenodo.org/badge/latestdoi/357330060)

## How to cite

If you use ShiftedProximalOperators.jl in your work, please cite using the format given in [CITATION.bib](CITATION.bib).

## Synopsis

ShiftedProximalOperators is a library of proximal operators associated with proper
lower-semicontinuous functions such as those implemented in
[ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
for use in the algorithms implemented in [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl).

The main difference between the proximal operators implemented in
[ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
is that those implemented here involve a translation of the nonsmooth term.
Specifically, this package considers proximal operators defined as

    argmin { ½ ‖t - q‖₂² + ν h(x + s + t) + χ(s + t; ΔB) | t ∈ ℝⁿ },

where q is given, x and s are fixed shifts, h is the nonsmooth term with respect
to which we are computing the proximal operator, and χ(.; ΔB) is the indicator of
a ball of radius Δ defined by a certain norm.

## How to Install

Until this package is registered, use
```julia
pkg> add https://github.com/rjbaraldi/ShiftedProximalOperators.jl
```

## What is Implemented?

Please refer to the documentation.

## Related Software

* [RegularizedProblems.jl](https://github.com/JuliaSmoothOptimizers/RegularizedProblems.jl)
* [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl)

## References

* A. Y. Aravkin, R. Baraldi and D. Orban, *A Proximal Quasi-Newton Trust-Region Method for Nonsmooth Regularized Optimization*, SIAM Journal on Optimization, 32(2), pp.900&ndash;929, 2022. Technical report: https://arxiv.org/abs/2103.15993

```bibtex
@article{aravkin-baraldi-orban-2022,
  author = {Aravkin, Aleksandr Y. and Baraldi, Robert and Orban, Dominique},
  title = {A Proximal Quasi-{N}ewton Trust-Region Method for Nonsmooth Regularized Optimization},
  journal = {SIAM Journal on Optimization},
  volume = {32},
  number = {2},
  pages = {900--929},
  year = {2022},
  doi = {10.1137/21M1409536},
  abstract = { We develop a trust-region method for minimizing the sum of a smooth term (f) and a nonsmooth term (h), both of which can be nonconvex. Each iteration of our method minimizes a possibly nonconvex model of (f + h) in a trust region. The model coincides with (f + h) in value and subdifferential at the center. We establish global convergence to a first-order stationary point when (f) satisfies a smoothness condition that holds, in particular, when it has a Lipschitz-continuous gradient, and (h) is proper and lower semicontinuous. The model of (h) is required to be proper, lower semi-continuous and prox-bounded. Under these weak assumptions, we establish a worst-case (O(1/\epsilon^2)) iteration complexity bound that matches the best known complexity bound of standard trust-region methods for smooth optimization. We detail a special instance, named TR-PG, in which we use a limited-memory quasi-Newton model of (f) and compute a step with the proximal gradient method, resulting in a practical proximal quasi-Newton method. We establish similar convergence properties and complexity bound for a quadratic regularization variant, named R2, and provide an interpretation as a proximal gradient method with adaptive step size for nonconvex problems. R2 may also be used to compute steps inside the trust-region method, resulting in an implementation named TR-R2. We describe our Julia implementations and report numerical results on inverse problems from sparse optimization and signal processing. Both TR-PG and TR-R2 exhibit promising performance and compare favorably with two linesearch proximal quasi-Newton methods based on convex models. }
}
```
