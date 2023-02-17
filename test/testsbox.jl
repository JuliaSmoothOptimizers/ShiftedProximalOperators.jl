for (op, shifted_op) ∈ zip((:NormL0, :NormL1), (:ShiftedNormL0Box, :ShiftedNormL1Box))
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
  end
end

for (op, shifted_op) ∈ zip((:NormL0, :NormL1), (:ShiftedNormL0Box, :ShiftedNormL1Box))
  @testset "$shifted_op iprox" begin
    if "$shifted_op" == "ShiftedNormL0Box"

      ## Testing ShiftedNormL0Box
      # We want to find argmin_t obj(t) = 1/2 * 1/σ * (t-q)^2 + λ * ||x+s+t||_0 + χ{s+t ∈ [l,u]} 
      # Parameters σ, l , u and s are fixed for sake of simplicity, only q, x and λ vary.

      # fixed parameters
      l = [-2.0] # l-s = -1
      u = [1.0] # u-s = 2
      s = [-1.0]

      # variable parameters (to cover different cases)
      d = [
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [2.0],
        [2.0],
        [2.0],
        [2.0],
        [2.0],
        [2.0],
        [-2.0],
        [-2.0],
        [-2.0],
      ]
      g = [
        [0.0],
        [0.0],
        [2.0],
        [2.0],
        [-2.0],
        [1.0],
        [0.0],
        [1.0],
        [10.0],
        [-10.0],
        [4.0],
        [-10.0],
        [10.0],
        [-4.0],
      ]
      x = [
        [0.0],
        [-10.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
      ]
      λ = [1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 0.1, 10.0, 1.0, 1.0, 10.0, 1.0, 1.0, 10.0]
      sol = [
        [1.0],
        [0.0],
        [-1.0],
        [1.0],
        [2.0],
        [-0.5],
        [0.0],
        [1.0],
        [-1.0],
        [2.0],
        [1.0],
        [2.0],
        [-1.0],
        [1.0],
      ]
      # Case 1 :  d = 0, g = 0 ; -(x+s) ∈ [l-s,u-s] -> solution : -x-s
      # Case 2 : d = 0, g = 0 ; -(x+s) ∉ [l-s,u-s] -> solution : 0
      # Case 3 : d = 0, g > 0 , -(x+s) ∈ [l-s,u-s], obj(-(x+s)) >= obj(l-s) -> solution : l-s
      # Case 4 : d = 0, g > 0 , -(x+s) ∈ [l-s,u-s], obj(-(x+s)) < obj(l-s) -> solution : -(x+s)
      # Case 5 : d = 0, g < 0 , -(x+s) ∈ [l-s,u-s], obj(-(x+s)) >= obj(u-s) -> solution : u-s
      # Case 6 : d > 0, -(g/d) ∈ [l-s,u-s], -(x+s) ∈ [l-s,u-s], obj(-(x+s)) > obj(-(g/d)) -> solution : -g/d
      # Case 7 : d > 0, -(g/d) = 0 ∈ [l-s,u-s], -(x+s) ∈ [l-s,u-s], obj(-(x+s)) > obj(-(g/d)) -> solution : 0
      # Case 8 : d > 0, -(g/d) ∈ [l-s,u-s], -(x+s) ∈ [l-s,u-s], obj(-(x+s)) < obj(-(g/d)) -> solution : -(x+s)
      # Case 9 : d > 0, -(g/d) < l-s, -(x+s) ∈ [l-s,u-s], obj(-(x+s)) > obj(l-s) -> solution : l-s
      # Case 10 : d > 0, -(g/d) > u-s, -(x+s) ∈ [l-s,u-s], obj(-(x+s)) > obj(u-s) -> solution : u-s
      # Case 11 : d > 0, -(g/d) < l-s, -(x+s) ∈ [l-s,u-s], obj(-(x+s)) < obj(l-s) -> solution : -(x+s)
      # Case 12 : d < 0, -(x+s) ∈ [l-s,u-s], obj(-(x+s)) > obj(l-s) > obj(u-s) -> solution : u-s
      # Case 13 : d < 0, -(x+s) ∈ [l-s,u-s], obj(-(x+s)) > obj(u-s) > obj(l-s) -> solution : l-s
      # Case 14 : d < 0, -(x+s) ∈ [l-s,u-s], obj(-(x+s)) < min(obj(l-s), obj(u-s)) -> solution : -(x+s)

    elseif "$shifted_op" == "ShiftedNormL1Box"

      ## Testing ShiftedNormL1Box
      # We want to find argmin obj(t) = (t-q)^2 + 2σλ||x+s+t||_1 + χ{s+t ∈ [l,u]} 
      # Parameters σ, l , u, λ and s are fixed for sake of simplicity, only q and x vary.

      # fixed parameters
      l = [-2.0] # l-s = -1
      u = [1.0] # u-s = 2
      s = [-1.0]

      # variable parameters (to cover different cases)
      d = [
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [2.0],
        [2.0],
        [2.0],
        [2.0],
        [2.0],
        [2.0],
        [-2.0],
        [-2.0],
        [-2.0],
      ]
      g = [
        [0.5],
        [0.5],
        [0.5],
        [2.0],
        [-2.0],
        [0.0],
        [1.0],
        [1.0],
        [-1.0],
        [1.0],
        [1.0],
        [0.0],
        [1.0],
        [1.0],
      ]
      x = [
        [0.0],
        [4.0],
        [-2.0],
        [0.0],
        [0.0],
        [4.0],
        [-2.0],
        [1.0],
        [0.5],
        [0.5],
        [3.0],
        [1.0],
        [1.0],
        [1.0],
      ]
      λ = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0]
      sol = [
        [1.0],
        [-1.0],
        [2.0],
        [-1.0],
        [2.0],
        [-0.5],
        [0.0],
        [0.0],
        [0.5],
        [0.0],
        [-1.0],
        [2.0],
        [0.0],
        [-1.0],
      ]
      # Case 1 : d = 0, |g| < λ, -(x+s) ∈ [l-s,u-s] -> solution : -x-s
      # Case 2 : d = 0, |g| < λ, -(x+s) < l-s -> solution : l-s
      # Case 3 : d = 0, |g| < λ, -(x+s) > u-s -> solution : u-s
      # Case 4 : d = 0, |g| > λ, g > 0 -> solution : l-s
      # Case 5 : d = 0, |g| > λ, g < 0 -> solution : u-s
      # Case 6 : d > 0, g = 0, -(x+s) < l-s, -(g+λ)/d ∈ [l-s, u-s] -> solution : -(g+λ)/d
      # Case 7 : d > 0, g != 0, -(x+s) > u-s, (-g+λ)/d ∈ [l-s, u-s] -> solution : -(g+λ)/d 
      # Case 8 : d > 0, g != 0, 
      #   -(g+λ)/d, (-g+λ)/d, -(x+s) ∈ [l-s, u-s], obj(-(g+λ)/d) lowest -> solution : -(g+λ)/d 
      # Case 9 : d > 0, g != 0, 
      #   -(g+λ)/d, (-g+λ)/d, -(x+s) ∈ [l-s, u-s], obj(-(x+s)) lowest -> solution : -(x+s)
      # Case 10 : d > 0, g != 0,
      #   -(g+λ)/d, (-g+λ)/d, -(x+s) ∈ [l-s, u-s], obj((-g+λ)/d) lowest -> solution : (-g+λ)/d
      # Case 11 : d > 0, g != 0, -(x+s) ∉ [l-s, u-s], obj(l-s) lowest -> solution : l - s
      # Case 12 : d < 0, g = 0, obj(u-s) < min(obj(l-s), obj(-(x+s))) -> solution : u-s
      # Case 13 : d < 0, g != 0, obj(-(x+s)) < min(obj(l-s), obj(u-s)) -> solution : -(x+s)
      # Case 14 : d < 0, g = 0, obj(l-s) < min(obj(u-s), obj(-(x+s))) -> solution : l-s
    end

    for i = 1:14
      # "$shifted_op" == "ShiftedNormL1Box" && i > 13 && continue
      gi = g[i]
      xi = x[i]
      di = d[i]
      λi = λ[i]
      h = eval(op)(λi)
      ψ = shifted(h, xi, l, u)
      ω = shifted(ψ, s)
      iprox(ω, gi, di)
      @test ω.sol == sol[i]
    end
  end
end
