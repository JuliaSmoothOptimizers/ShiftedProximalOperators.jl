using ShiftedProximalOperators
using Test


"""
Testing ShiftedNormL0Box

We want to find argmin obj(t) = 1/2 * 1/σ * (t-q)^2 + λ||x+s+t||_0 + χ{s+t ∈ [l,u]} 
Parameters σ, l , u and s are fixed for sake of simplicity, only q, x and λ vary.
"""

σ = 1.0 ; l = [0.0] ; u = [3] ; s = [-1.0] # fixed parameters

qxλ = # variable parameters
[5.0 1.0 1.0; # q>u-s ; -(x+s)∉[l-s,u-s] -> solution : u-s
5.0 -1.0 5.0; # q>u-s , -(x+s)∈[l-s,u-s], obj(-(x+s)) < obj(u-s) -> solution : -(x+s)
5.0 -1.0 3.0; # q>u-s, -(x+s)∈[l-s,u-s], obj(-(x+s)) >= obj(u-s) -> solution : u-s
0.0 1.0 1.0; # q<l-s, -(x+s)∉[l-s,u-s] -> solution : l-s
0.0 -1.0 2.0; # q<l-s, -(x+s)∈[l-s,u-s], obj(-(x+s)) < obj(l-s) -> solution : -(x+s)
0.0 -1.0 1.0; # q<l-s, -(x+s)∈[l-s,u-s], obj(-(x+s)) >= obj(l-s) -> solution : l-s
3.0 1.0 1.0; # q∈[l-s,u-s], -(x+s)∉[l-s,u-s] -> solution : q
3.0 -1.0 1.0; # q∈[l-s,u-s], -(x+s)∈[l-s,u-s], obj(-(x+s)) < obj(q) -> solution : -(x+s)
3.0 -1.0 0.1] # q∈[l-s,u-s], -(x+s)∈[l-s,u-s], obj(-(x+s)) >= obj(q) -> solution : q

sol = [4.0, 2.0, 4.0, 1.0, 2.0, 1.0, 3.0, 2.0, 3.0]

for i in 1:9
  q = [qxλ[i,1]] ; x = [qxλ[i,2]] ; λ = qxλ[i,3]
  h = NormL0(λ) ; ψ = shifted(h,x,l,u) ; ω = shifted(ψ,s)
  ShiftedProximalOperators.prox(ω, q, σ)
  @test ω.sol == [sol[i]]
end


"""
Testing ShiftedNormL0Box

We want to find argmin obj(t) = 1/2 * 1/σ * (t-q)^2 + λ||x+s+t||_0 + χ{s+t ∈ [l,u]} 
Parameters σ, l , u and s are fixed for sake of simplicity, only q, x and λ vary.
"""

σ = 1.0 ; l = [0.0] ; u = [3] ; s = [-1.0] # fixed parameters

qxλ = # variable parameters
[5.0 1.0 1.0; # q>u-s ; -(x+s)∉[l-s,u-s] -> solution : u-s
5.0 -1.0 5.0; # q>u-s , -(x+s)∈[l-s,u-s], obj(-(x+s)) < obj(u-s) -> solution : -(x+s)
5.0 -1.0 3.0; # q>u-s, -(x+s)∈[l-s,u-s], obj(-(x+s)) >= obj(u-s) -> solution : u-s
0.0 1.0 1.0; # q<l-s, -(x+s)∉[l-s,u-s] -> solution : l-s
0.0 -1.0 2.0; # q<l-s, -(x+s)∈[l-s,u-s], obj(-(x+s)) < obj(l-s) -> solution : -(x+s)
0.0 -1.0 1.0; # q<l-s, -(x+s)∈[l-s,u-s], obj(-(x+s)) >= obj(l-s) -> solution : l-s
3.0 1.0 1.0; # q∈[l-s,u-s], -(x+s)∉[l-s,u-s] -> solution : q
3.0 -1.0 1.0; # q∈[l-s,u-s], -(x+s)∈[l-s,u-s], obj(-(x+s)) < obj(q) -> solution : -(x+s)
3.0 -1.0 0.1] # q∈[l-s,u-s], -(x+s)∈[l-s,u-s], obj(-(x+s)) >= obj(q) -> solution : q

sol = [4.0, 2.0, 4.0, 1.0, 2.0, 1.0, 3.0, 2.0, 3.0]

for i in 1:9
  q = [qxλ[i,1]] ; x = [qxλ[i,2]] ; λ = qxλ[i,3]
  h = NormL0(λ) ; ψ = shifted(h,x,l,u) ; ω = shifted(ψ,s)
  ShiftedProximalOperators.prox(ω, q, σ)
  @test ω.sol == [sol[i]]
end