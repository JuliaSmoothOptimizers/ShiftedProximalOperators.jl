for (op, shifted_op) ∈ zip((:NormL0, :NormL1), (:ShiftedNormL0Box, :ShiftedNormL1Box))
  @testset "$shifted_op" begin
    if "$shifted_op" == "ShiftedNormL0Box"

      ## Testing ShiftedNormL0Box
      # We want to find argmin_t obj(t) = 1/2 * 1/σ * (t-q)^2 + λ * ||x+s+t||_0 + χ{s+t ∈ [l,u]} 
      # Parameters σ, l , u and s are fixed for sake of simplicity, only q, x and λ vary.

      # fixed parameters
      σ = 1.0
      d = [1.0]
      d2 = [-1.0]
      d3 = [0.0]
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

      # Cases where d = [-1.0] for iprox
      q2 = [[5.0], [5.0], [5.0], [0.0], [0.0], [0.0], [3.0], [3.0], [2.0]]
      x2 = [[1.0], [-1.0], [-1.0], [1.0], [-1.0], [-1.0], [-1.0], [-1.0], [-1.0]]
      λ2 = [1.0, 5.0, 3.0, 1.0, 7.0, 1.0, 2.0, 1.0, 1.0]
      sol2 = [[1.0], [2.0], [1.0], [4.0], [2.0], [4.0], [2.0], [1.0], [4.0]]
      # Case 1 : q > u-s ; -(x+s) ∉ [l-s,u-s] -> solution : l-s
      # Case 2 : q > u-s , -(x+s) ∈ [l-s,u-s], obj(-(x+s)) < obj(l-s) -> solution : -(x+s)
      # Case 3 : q > u-s, -(x+s) ∈ [l-s,u-s], obj(-(x+s)) >= obj(l-s) -> solution : l-s
      # Case 4 : q < l-s, -(x+s) ∉ [l-s,u-s] -> solution : u-s
      # Case 5 : q < l-s, -(x+s) ∈ [l-s,u-s], obj(-(x+s)) < obj(u-s) -> solution : -(x+s)
      # Case 6 : q < l-s, -(x+s) ∈ [l-s,u-s], obj(-(x+s)) >= obj(u-s) -> solution : u-s
      # Case 7 : q ∈ [l-s,u-s], -(x+s) ∈ [l-s,u-s], obj(-(x+s)) < min(obj(l-s), obj(u-s)) -> solution : -(x+s)
      # Case 8 : q ∈ [l-s,u-s], -(x+s) ∈ [l-s,u-s], min(obj(-(x+s)), obj(u-s)) >= obj(l-s) -> solution : l-s
      # Case 9 : q ∈ [l-s,u-s], -(x+s) ∈ [l-s,u-s], min(obj(-(x+s)), obj(l-s)) >= obj(u-s) -> solution : u-s

      # Cases where d = [0.0] for iprox
      q3 = [[5.0], [5.0]]
      x3 = [[1.0], [-1.0]]
      λ3 = [1.0, 1.0]
      sol3 = [[0.0], [2.0]]
      # Case 1 : -(x+s) ∉ [l-s,u-s] -> solution : 0
      # Case 2 : -(x+s) ∈ [l-s,u-s] -> solution : -(x+s)

    elseif "$shifted_op" == "ShiftedNormL1Box"

      ## Testing ShiftedNormL1Box
      # We want to find argmin obj(t) = (t-q)^2 + 2σλ||x+s+t||_1 + χ{s+t ∈ [l,u]} 
      # Parameters σ, l , u, λ and s are fixed for sake of simplicity, only q and x vary.

      # fixed parameters
      σ = 1.0
      d = [1.0]
      d2 = [-1.0]
      d3 = [0.0]
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

      # Cases where d = [-1.0] for iprox
      q2 = [[-5.0], [5.5], [3.0], [-2.0], [7.0], [7.0], [-1.0]]
      x2 = [[-4.0], [-4.0], [-2.0], [-2.0], [-5.0], [1.0], [3.0]]
      sol2 = [[4.0], [1.0], [3.0], [4.0], [1.0], [1.0], [4.0]]
      # Case 1 : -(x+s) > u-s   q+λdi < l-s -> solution : u-s
      # Case 2 : -(x+s) > u-s   -(x+s) > q+λdi > u-s -> solution : l-s
      # Case 3 : l-s <-(x+s) < u-s   big λ (directly in test below) -> solution : -x-s
      # Case 4 : l-s <-(x+s) < u-s   q+λdi < l-s, small λ (1.0) -> solution : u-s
      # Case 5 : l-s <-(x+s) < u-s   q+λdi > u-s, small λ (1.0) -> solution : l-s
      # Case 6 : l-s >-(x+s)   q+λdi > u-s -> solution : l-s
      # Case 7 : l-s >-(x+s)   -(x+s) < q+λdi < l-s -> solution : u-s

      # Cases where d = [0.0] for iprox
      q3 = [[5.0], [5.0], [5.0]]
      x3 = [[1.0], [-1.0], [-4.0]]
      sol3 = [[1.0], [2.0], [4.0]]
      # Case 1 : -(x+s) < l-s -> solution : l-s
      # Case 2 : -(x+s) ∈ [l-s,u-s] -> solution : -(x+s)
      # Case 2 : -(x+s) > u-s -> solution : u-s
    end

    for i = 1:9
      qi = q[i]
      xi = x[i]
      "$shifted_op" == "ShiftedNormL0Box" ? λi = λ[i] : λi = λ
      h = eval(op)(λi)
      ψ = shifted(h, xi, l, u)
      ω = shifted(ψ, s)
      ShiftedProximalOperators.prox(ω, qi, σ)
      @test ω.sol == sol[i]
    end
    # iprox, d = 1.0
    for i = 1:9
      qi = q[i]
      xi = x[i]
      "$shifted_op" == "ShiftedNormL0Box" ? λi = λ[i] : λi = λ
      h = eval(op)(λi)
      ψ = shifted(h, xi, l, u)
      ω = shifted(ψ, s)
      ShiftedProximalOperators.iprox(ω, qi, d)
      @test ω.sol == sol[i]
    end
    # iprox, d = -1.0
    for i=1:9
      "$shifted_op" == "ShiftedNormL1Box" && i > 7 && continue
      qi2 = q2[i]
      xi2 = x2[i]
      "$shifted_op" == "ShiftedNormL0Box" ? λi2 = λ2[i] : (λi2 = (i == 3) ? 10.0 : λ)
      h = eval(op)(λi2)
      ψ = shifted(h, xi2, l, u)
      ω = shifted(ψ, s)
      ShiftedProximalOperators.iprox(ω, qi2, d2)
      @test ω.sol == sol2[i]
    end
    # iprox, d = 0.0 
    for i=1:3
      "$shifted_op" == "ShiftedNormL0Box" && i > 2 && continue
      qi3 = q3[i]
      xi3 = x3[i]
      "$shifted_op" == "ShiftedNormL0Box" ? λi3 = λ3[i] : λi3 = λ
      h = eval(op)(λi3)
      ψ = shifted(h, xi3, l, u)
      ω = shifted(ψ, s)
      ShiftedProximalOperators.iprox(ω, qi3, d3)
      @test ω.sol == sol3[i]
    end
  end
end
