# ShiftedProximalOperators

[![CI](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaSmoothOptimizers/ShiftedProximalOperators.jl/actions/workflows/ci.yml)
[![](https://img.shields.io/badge/docs-latest-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/ShiftedProximalOperators.jl/dev)
[![codecov](https://codecov.io/gh/JuliaSmoothOptimizers/ShiftedProximalOperators.jl/branch/master/graph/badge.svg?token=CZzi6ufcXI)](https://codecov.io/gh/JuliaSmoothOptimizers/ShiftedProximalOperators.jl)

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

* A. Y. Aravkin, R. Baraldi and D. Orban, *A Proximal Quasi-Newton Trust-Region Method for Nonsmooth Regularized Optimization*, Cahier du GERAD G-2021-12, GERAD, Montréal, Canada. https://arxiv.org/abs/2103.15993

