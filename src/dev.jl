
### Des commandes pour tester la lib

#=
h = NormL1(1.0)
n = 4
Δ = 2 * rand()
q = 2 * (rand(n) .- 0.5)
χ = NormLinf(1.0)
ν = rand()
xk = rand(n) .- 0.5
ψ = shifted(h, xk, Δ, χ) # idee : shifted(h, xk, l, u, χ)
ShiftedProximalOperators.prox(ψ, q, ν)
=#


### Pb : on fait appel a IndBallLinf qui est une fonction de la lib ProximalOperators
### or cela correspond uniquement à une indicatrice pour une region de la forme [-a,a]
### alors aue l'on voudrait [l,u] quelconque. la fonction ci dessous implemente ca (IndSet).

using LinearAlgebra
using ProximalOperators


struct IndSet
    l::Real
    u::Real
    IndSet(l,u) = l > u ? error("out of order") : new(l,u)
end


function (f::IndSet)(x)
    ind = zeros(Float64,size(x))
    ind[.!(f.l .<= x .<= f.u)] .= Inf
    return ind
end





