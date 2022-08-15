using ProximalOperators
using ShiftedProximalOperators
using Test

for (op, shifted_op) ∈ zip((:NormL0, :NormL1),(:ShiftedNormL0Box, :ShiftedNormL1Box))
  @testset "$shifted_op" begin
    if "$shifted_op" == "ShiftedNormL0Box"

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

    elseif "$shifted_op" == "ShiftedNormL1Box"

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
      q = [[0.5], [5.0], [3.0], [-2.0], [4.0], [1.0], [1.0], [7.0], [4.0]]
      x = [[1.0], [-4.0], [-2.0], [-1.0], [-5.0], [-3.0], [3.0], [-2.0], [1.0]]
      sol = [[1.0], [4.0], [3.0], [1.0], [4.0], [2.0], [1.0], [4.0], [3.0]]
      # Case 1 : q-σλ < -(x+s) < q+σλ ; -x < l -> solution : l-s
      # Case 2 : q-σλ < -(x+s) < q+σλ ; -x > u -> solution : u-s
      # Case 3 : q-σλ < -(x+s) < q+σλ ; l < -x < u -> solution : -(x+s)
      # Case 4 : q+σλ < -(x+s) ; q+σλ < l-s -> solution : l-s
      # Case 5 : q+σλ < -(x+s) ; q+σλ > u-s -> solution : u-s
      # Case 6 : q+σλ < -(x+s) ; l-s < q+σλ < u-s -> solution : q+σλ
      # Case 7 : q-σλ > -(x+s) ; q-σλ < l-s -> solution : l-s
      # Case 8 : q-σλ > -(x+s) ; q-σλ > u-s -> solution : u-s
      # Case 9 : q-σλ > -(x+s) ; l-s < q-σλ < u-s -> solution : q-σλ
    end

    for i in 1:9
      qi = q[i]
      xi = x[i]
      "$shifted_op" == "ShiftedNormL0Box" ? λi = λ[i] : λi = λ
      h = eval(op)(λi)
      ψ = shifted(h,xi,l,u,1.0)
      ω = shifted(ψ,s)
      ShiftedProximalOperators.prox(ω, qi, σ)
      @test ω.sol == sol[i]
    end
  end
end
