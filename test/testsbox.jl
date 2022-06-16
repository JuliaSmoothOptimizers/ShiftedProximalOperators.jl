using ProximalOperators
using ShiftedProximalOperators
using Test


## Testing ShiftedNormL0Box
# We want to find argmin_t obj(t) = 1/2 * 1/σ * (t-q)^2 + λ * ||x+s+t||_0 + χ{s+t ∈ [l,u]} 
# Parameters σ, l , u and s are fixed for sake of simplicity, only q, x and λ vary.

# fixed parameters
σ = 1.0 
l = [0.0]
u = [3.0]
s = [-1.0]

# variable parameters (to cover the 9 different cases)
q = [[5.0], [5.0], [5.0], [0.0], [0.0], [0.0], [3.0], [3.0], [3.0]]
x = [[1.0], [-1.0], [-1.0], [1.0], [-1.0], [-1.0], [1.0], [-1.0], [-1.0]]
λ = [1.0, 5.0, 3.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.1]
sol = [[4.0], [2.0], [4.0], [1.0], [2.0], [1.0], [3.0], [2.0], [3.0]]
# Case 1 : q > u-s ; -(x+s) ∉ [l-s,u-s] -> solution : u-s
# Case 2 : q > u-s , -(x+s) ∈ [l-s,u-s], obj(-(x+s)) < obj(u-s) -> solution : -(x+s)
# Case 3 : q > u-s, -(x+s) ∈ [l-s,u-s], obj(-(x+s)) >= obj(u-s) -> solution : u-s
# Case 4 : q < l-s, -(x+s) ∉ [l-s,u-s] -> solution : l-s
# Case 5 : q < l-s, -(x+s) ∈ [l-s,u-s], obj(-(x+s)) < obj(l-s) -> solution : -(x+s)
# Case 6 : q < l-s, -(x+s) ∈ [l-s,u-s], obj(-(x+s)) >= obj(l-s) -> solution : l-s
# Case 7 : q ∈ [l-s,u-s], -(x+s) ∉ [l-s,u-s] -> solution : q
# Case 8 : q ∈ [l-s,u-s], -(x+s) ∈ [l-s,u-s], obj(-(x+s)) < obj(q) -> solution : -(x+s)
# Case 9 : q ∈ [l-s,u-s], -(x+s) ∈ [l-s,u-s], obj(-(x+s)) >= obj(q) -> solution : q

for i in 1:9
  qi = q[i]
  xi = x[i]
  λi = λ[i]
  h = NormL0(λi)
  ψ = shifted(h,xi,l,u)
  ω = shifted(ψ,s)
  ShiftedProximalOperators.prox(ω, qi, σ)
  @test ω.sol == sol[i]
end



## Testing ShiftedNormL1Box
# We want to find argmin obj(t) = (t-q)^2 + 2σλ||x+s+t||_1 + χ{s+t ∈ [l,u]} 
# Parameters σ, l , u, λ and s are fixed for sake of simplicity, only q and x vary.

# fixed parameters
σ = 1.0
l = [0.0]
u = [3.0]
s = [-1.0]
λ = 1.0 

# variable parameters (to cover the 9 different cases)
q = [[1.0], [5.0], [3.0], [-2.0], [3.0], [1.0], [1.0], [7.0], [4.0]]
x = [[1.0], [-4.0], [-2.0], [-1.0], [-5.0], [-3.0], [3.0], [-2.0], [1.0]]
sol = [[1.0], [4.0], [3.0], [1.0], [4.0], [3.0], [1.0], [4.0], [2.0]]
# Case 1 : q-2σλ < -(x+s) < q+2σλ ; -x < l -> solution : l-s
# Case 2 : q-2σλ < -(x+s) < q+2σλ ; -x > u -> solution : u-s
# Case 3 : q-2σλ < -(x+s) < q+2σλ ; l < -x < u -> solution : -(x+s)
# Case 4 : q+2σλ < -(x+s) ; q+2σλ < l-s -> solution : l-s
# Case 5 : q+2σλ < -(x+s) ; q+2σλ > u-s -> solution : u-s
# Case 6 : q+2σλ < -(x+s) ; l-s < q+2σλ < u-s -> solution : q+2σλ
# Case 7 : q-2σλ > -(x+s) ; q-2σλ < l-s -> solution : l-s
# Case 8 : q-2σλ > -(x+s) ; q-2σλ > u-s -> solution : u-s
# Case 9 : q-2σλ > -(x+s) ; l-s < q-2σλ < u-s -> solution : q-2σλ


for i in 1:9
  qi = q[i]
  xi = x[i]
  h = NormL1(λ)
  ψ = shifted(h,xi,l,u)
  ω = shifted(ψ,s)
  ShiftedProximalOperators.prox(ω, qi, σ)
  @test ω.sol == sol[i]
end