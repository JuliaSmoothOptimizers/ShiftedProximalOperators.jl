# ShiftedProximalOperators.jl

## Introduction

ShiftedProximalOperators is a library of proximal operators associated with proper
lower-semicontinuous functions such as those implemented in
[ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
for use in the algorithms implemented in [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl).

The main differences with the proximal operators implemented in
[ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
are that those implemented here involve a shift of the nonsmooth term and may include an extra indicator function.
We also implement new proximal operators.

## Proximal operators

The operators from 
[ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)
can be written as

```math
\mathrm{prox}_{\nu h}(q) = \arg\min_t \left\{ \tfrac{1}{2}\|t-q\|^2 + \nu h(t) \right\}
```

We consider a proximal operator from [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl), for example [`ProximalOperators.NormL1`](https://juliafirstorder.github.io/ProximalOperators.jl/stable/functions/#ProximalOperators.NormL1).

```julia
using ProximalOperators, ShiftedProximalOperators
h = NormL1(1.0) # proximal operator
```

This package provides the following shifted proximal operators,
where `x` and `s` are fixed shifts, `h` is the nonsmooth term with respect
to which we are computing the proximal operator.

## Basic shifted proximal operator

```julia
ψ = shifted(h, x)
```

models

```math
\mathrm{prox}_{\nu h}(q; x) = \arg\min_t \left\{ \tfrac{1}{2}\|t-q\|^2 + \nu h(x + t) \right\}
```

`ψ` can be shifted again using

```julia
ψs = shifted(ψ, sj)
```

which models

```math
\mathrm{prox}_{\nu h}(q; x, s) = \arg\min_t \left\{ \tfrac{1}{2}\|t-q\|^2 + \nu h(x + s + t) \right\}
```

## Ball shifted proximal operator

Let `χ(.; ΔB)` be the indicator of a ball of radius `Δ` defined by a certain norm.

```julia
χ = NormL2(1.0) # choose the 2-norm for χ
ψ = shifted(h, x, Δ, χ) # Δ is the radius of the ball ΔB
```

models

```math
\mathrm{prox}_{\nu h + \chi}(q; x) = \arg\min_t \left\{ \tfrac{1}{2}\|t-q\|^2 + \nu h(x + t) + \chi (t; \Delta \mathcal{B}) \right\}
```

`ψ` can be shifted again using

```julia
ψs = shifted(ψ, sj)
```

which models

```math
\mathrm{prox}_{\nu h + \chi}(q; x, s) = \arg\min_t \left\{ \tfrac{1}{2}\|t-q\|^2 + \nu h(x + s + t) + \chi (s + t; \Delta \mathcal{B}) \right\}
```

## Box shifted proximal operator

```julia
ψ = shifted(h, x, l, u)
```

models

```math
\mathrm{prox}_{\nu h + \chi}(q; x) = \arg\min_t \left\{ \tfrac{1}{2}\|t-q\|^2 + \nu h(x + t) + \chi (t; [\ell, u]) \right\}
```

where `χ(t; [ℓ, u])` is `0` if `tᵢ ∈ [ℓᵢ, uᵢ]` for all `i ∈ [1, length(t)]` and `+∞` otherwise.
`ψ` can be shifted again using

```julia
ψs = shifted(ψ, sj)
```

which models

```math
\mathrm{prox}_{\nu h + \chi}(q; x, s) = \arg\min_t \left\{ \tfrac{1}{2}\|t-q\|^2 + \nu h(x + s + t) + \chi (s + t; [\ell, u]) \right\}
```
