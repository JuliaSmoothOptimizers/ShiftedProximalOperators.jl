using ProximalOperators
using ShiftedProximalOperators
using Test

"""
Testing ShiftedNormL0Box

We want to find argmin_t obj(t) = 1/2 * 1/σ * (t-q)^2 + λ * I{x+s+t ≂̸ 0} + χ{s+t ∈ [l,u]} 
Parameters σ, l , u and s are fixed for sake of simplicity, only q, x and λ vary.
"""

σ = 1.0 ; l = [0.0] ; u = [3.0] ; s = [-1.0] # fixed parameters

qxλ = # variable parameters (q in first column, x in second column, λ in third column)
[5.0 1.0 1.0; # q > u-s ; -(x+s) ∉ [l-s,u-s] -> solution : u-s
5.0 -1.0 5.0; # q > u-s , -(x+s) ∈ [l-s,u-s], obj(-(x+s)) < obj(u-s) -> solution : -(x+s)
5.0 -1.0 3.0; # q > u-s, -(x+s) ∈ [l-s,u-s], obj(-(x+s)) >= obj(u-s) -> solution : u-s
0.0 1.0 1.0; # q < l-s, -(x+s) ∉ [l-s,u-s] -> solution : l-s
0.0 -1.0 2.0; # q < l-s, -(x+s) ∈ [l-s,u-s], obj(-(x+s)) < obj(l-s) -> solution : -(x+s)
0.0 -1.0 1.0; # q < l-s, -(x+s) ∈ [l-s,u-s], obj(-(x+s)) >= obj(l-s) -> solution : l-s
3.0 1.0 1.0; # q ∈ [l-s,u-s], -(x+s) ∉ [l-s,u-s] -> solution : q
3.0 -1.0 1.0; # q ∈ [l-s,u-s], -(x+s) ∈ [l-s,u-s], obj(-(x+s)) < obj(q) -> solution : -(x+s)
3.0 -1.0 0.1] # q ∈ [l-s,u-s], -(x+s) ∈ [l-s,u-s], obj(-(x+s)) >= obj(q) -> solution : q

sol = [4.0, 2.0, 4.0, 1.0, 2.0, 1.0, 3.0, 2.0, 3.0]

for i in 1:9
  q = [qxλ[i,1]] ; x = [qxλ[i,2]] ; λ = qxλ[i,3]
  h = NormL0(λ) ; ψ = shifted(h,x,l,u) ; ω = shifted(ψ,s)
  ShiftedProximalOperators.prox(ω, q, σ)
  @test ω.sol == [sol[i]]
end


"""
Testing ShiftedNormL1Box

We want to find argmin obj(t) = (t-q)^2 + 2σλ|x+s+t| + χ{s+t ∈ [l,u]} 
Parameters σ, l , u, λ and s are fixed for sake of simplicity, only q and x vary.
"""

σ = 1.0 ; l = [0.0] ; u = [3.0] ; s = [-1.0] ; λ = 1.0 # fixed parameters

qx = # variable parameters
[1.0 1.0; # q-2σλ < -(x+s) < q+2σλ ; -x < l -> solution : l-s
5.0 -4.0; # q-2σλ < -(x+s) < q+2σλ ; -x > u -> solution : u-s
3.0 -2.0; # q-2σλ < -(x+s) < q+2σλ ; l < -x < u -> solution : -(x+s)
-2.0 -1.0; # q+2σλ < -(x+s) ; q+2σλ < l-s -> solution : l-s
3.0 -5.0; # q+2σλ < -(x+s) ; q+2σλ > u-s -> solution : u-s
1.0 -3.0; # q+2σλ < -(x+s) ; l-s < q+2σλ < u-s -> solution : q+2σλ
1.0 3.0; # q-2σλ > -(x+s) ; q-2σλ < l-s -> solution : l-s
7.0 -2.0; # q-2σλ > -(x+s) ; q-2σλ > u-s -> solution : u-s
4.0 1.0] # q-2σλ > -(x+s) ; l-s < q-2σλ < u-s -> solution : q-2σλ

sol = [1.0, 4.0, 3.0, 1.0, 4.0, 3.0, 1.0, 4.0, 2.0]

for i in 1:9
  q = [qx[i,1]] ; x = [qx[i,2]]
  h = NormL1(λ) ; ψ = shifted(h,x,l,u) ; ω = shifted(ψ,s)
  ShiftedProximalOperators.prox(ω, q, σ)
  @test ω.sol == [sol[i]]
end