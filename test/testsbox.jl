for (op, shifted_op) ∈ zip(
  (:NormL0, :NormL1, :RootNormLhalf),
  (:ShiftedNormL0Box, :ShiftedNormL1Box, :ShiftedRootNormLhalfBox),
)
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

    elseif "$shifted_op" == "ShiftedRootNormLhalfBox"

      ## Testing ShiftedRootNormLhalfBox
      # We want to find argmin_t obj(t) = 1/2 * 1/σ * (t-q)^2 + λ * √(|x+s+t|) + χ{s+t ∈ [l,u]} 
      # Parameters σ, l , u and s are fixed for sake of simplicity, only q, x and λ vary.

      # fixed parameters
      σ = 1.0
      l = [0.0]
      u = [3.0]
      s = [-1.0]

      # variable parameters (to cover the 9 different cases)
      q = [[5.0], [5.0], [5.0], [2.0], [0.0], [1.0], [0.0], [3.0], [3.0]]
      x = [[1.0], [-1.0], [-1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0], [-1.0]]
      λ = [1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0]
      sol = [[4.0], [2.0], [4.0], [1.6054], [1.0], [2.0], [1.0], [2.702], [2.0]]
      # Case 1 : q > u-s ; -(x+s), val ∉ [l-s,u-s] -> solution : u-s
      # Case 2 : q > u-s , -(x+s), val ∈ [l-s,u-s], obj(-(x+s)) < obj(u-s) -> solution : -(x+s)
      # Case 3 : q > u-s, -(x+s), val ∈ [l-s,u-s], obj(-(x+s)) >= obj(u-s) -> solution : u-s
      # Case 4 : -(x+s) ∉ [l-s,u-s], val ∈ [l-s,u-s], obj(val) smallest -> solution : val
      # Case 5 : q < l-s, -(x+s), val ∉ [l-s,u-s], obj(-(x+s)) >= obj(l-s) -> solution : l-s
      # Case 6 : q < l-s, -(x+s) ∈ [l-s,u-s], val ∉ [l-s,u-s], obj(-(x+s)) < obj(l-s) -> solution : -(x+s)
      # Case 7 : q < l-s, -(x+s) ∈ [l-, u-s], val ∉ [l-s,u-s], obj(-(x+s)) >= obj(l-s) -> solution : l-s
      # Case 8 : -(x+s), val ∈ [l-s,u-s], obj(val) smallest -> solution : val
      # Case 9 : -(x+s), val ∈ [l-s,u-s], obj(-(x+s)) smallest -> solution : -(x+s)
    end

    for i = 1:9
      qi = q[i]
      xi = x[i]
      "$shifted_op" == "ShiftedNormL1Box" ? λi = λ : λi = λ[i]
      h = eval(op)(λi)
      ψ = shifted(h, xi, l, u)
      ω = shifted(ψ, s)
      ShiftedProximalOperators.prox(ω, qi, σ)
      @test isapprox(ω.sol, sol[i], atol = 1.0e-2) # for approximate value val with RootNormLhalf
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

for (op, shifted_op) ∈ zip((:GroupNormL2,), (:ShiftedGroupNormL2Box,))
  @testset "$shifted_op" begin
    ## Testing ShiftedGroupNormL2Box
    # Looking for argmin_t obj(t) = 1/(2σ) * ‖t-q‖² + λ * ‖x+s+t‖₂ + χ{s+t ∈ [l,u]}

    σ = 1.0
    λ = [1.0]
    idx = [1:2]
    l = [-1.0, -1.0]
    u = [1.0, 1.0]
    s = [0.0, 0.0]

    x = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5, 0.5]]
    q = [[3.0, 4.0], [0.3, 0.4], [-3.0, -4.0], [2.5, 3.5]]
    sol = [[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [1.0, 1.0]]
    # Case 1 : block soft-threshold out of the box -> clip u-s
    # Case 2 : ‖q+x+s‖ ≤ σλ -> prox = 0, in the box -> 0
    # Case 3 : block soft-threshold out of the box -> clip l-s
    # Case 4 : xk ≠ 0, solution out of the box -> clip u-s

    for i = 1:4
      h = GroupNormL2(λ, idx)
      ψ = shifted(h, x[i], l, u)
      ω = shifted(ψ, s)
      y = similar(q[i])
      prox!(y, ω, q[i], σ)
      @test isapprox(y, sol[i], atol = 1.0e-8)
    end
  end

  @testset "ShiftedGroupNormL2Box with selected" begin
    σ = 1.0
    λ = [1.0]
    idx = [1:3]
    l = [-1.0, -1.0, -1.0]
    u = [1.0, 1.0, 1.0]
    s = [0.0, 0.0, 0.0]
    x = [0.0, 0.0, 0.0]
    selected = [1, 2]   # index 3 excluded from the group penalty

    q = [[3.0, 4.0, 0.0], [3.0, 4.0, 5.0], [3.0, 4.0, -5.0]]
    sol = [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0]]
    # Case 1 : q[3] = 0 ∈ [l-s,u-s] -> prox_zero(0,-1,1) = 0
    # Case 2 : q[3] = 5 > u-s = 1   -> prox_zero(5,-1,1) = 1 (clip u-s)
    # Case 3 : q[3] = -5 < l-s = -1 -> prox_zero(-5,-1,1) = -1 (clip l-s)

    for i = 1:3
      h = GroupNormL2(λ, idx)
      ψ = shifted(h, x, l, u, selected)
      ω = shifted(ψ, s)
      y = similar(q[i])
      prox!(y, ω, q[i], σ)
      @test isapprox(y, sol[i], atol = 1.0e-8)
    end

    # check if selected is propagated after the shift
    h = GroupNormL2(λ, idx)
    ψ = shifted(h, x, l, u, selected)
    ω = shifted(ψ, s)
    @test ω.selected == selected
  end
end

for (op, shifted_op) ∈ zip((:NormL2Inf,), (:ShiftedNormL2InfBox,))
  @testset "$shifted_op" begin
    ## Testing ShiftedNormL2InfBox
    # Looking for argmin_t  1/(2σ) ‖t-q‖² + max_i ωᵢ ‖xk+sj+t‖² + χ{sj+t ∈ [l,u]}

    sigma = 1.0
    omega = [1.0, 2.0]
    n_group = 2
    l = [-1.0, -1.0, -1.0, -1.0]
    u = [ 1.0,  1.0,  1.0,  1.0]
    s = [0.0, 0.0, 0.0, 0.0]

    x = [
      [ 0.0,  0.0,  0.0,  0.0],   # cas 1 : q grand → prox >> u, clip u pour les 2 groupes
      [ 0.0,  0.0,  0.0,  0.0],   # cas 2 : q petit négatif → prox << l, clip l
      [ 0.0,  0.0,  0.0,  0.0],   # cas 3 : q petit → prox dans la boîte, vérif sans clip
      [ 0.5,  0.5,  0.5,  0.5],   # cas 4 : xk ≠ 0, v très grand → clip u pour les 2 groupes
    ]
    q = [
      [ 3.0,  4.0,  3.0,  4.0],
      [-3.0, -4.0, -3.0, -4.0],
      [ 0.3,  0.3,  0.3,  0.3],
      [ 4.5,  4.5,  4.5,  4.5],   # v = q+xk = [5,5,5,5] → prox >> u pour les 2 groupes
    ]
    sol = [
      [ 1.0,  1.0,  1.0,  1.0],   # cas 1 : clip u
      [-1.0, -1.0, -1.0, -1.0],   # cas 2 : clip l
      nothing,                     # cas 3 : vérifié autrement
      [ 1.0,  1.0,  1.0,  1.0],   # cas 4 : clip u
    ]

    for i in [1, 2, 4]
      h = NormL2Inf(omega, n_group)
      ψ = shifted(h, x[i], l, u)
      ω_op = shifted(ψ, s)
      y = similar(q[i])
      prox!(y, ω_op, q[i], sigma)
      @test isapprox(y, sol[i], atol = 1e-8)
    end

    # cas 3 : prox dans la boîte → résultat identique au prox non contraint
    let i = 3
      h = NormL2Inf(omega, n_group)
      ψ_box = shifted(h, x[i], l, u)
      ψ_unc = shifted(h, x[i])
      ω_box = shifted(ψ_box, s)
      ω_unc = shifted(ψ_unc, s)
      y_box = similar(q[i])
      y_unc = similar(q[i])
      prox!(y_box, ω_box, q[i], sigma)
      prox!(y_unc, ω_unc, q[i], sigma)
      @test all(y_box .≥ l .- 1e-10)
      @test all(y_box .≤ u .+ 1e-10)
      @test isapprox(y_box, y_unc, atol = 1e-8)
    end

    h = NormL2Inf(omega, n_group)
    ψ = shifted(h, zeros(4), l, u)
    @test ψ(2 .* ones(4)) == Inf

    ψ2 = shifted(h, zeros(4), l, u)
    @test all(ψ2.sj .== 0)
    @test all(ψ2.xk .== 0)
    @test ψ2.l == l
    @test ψ2.u == u
    @test ψ2.selected == 1:4

    s2 = 0.5 * ones(4)
    φ2 = shifted(ψ2, s2)
    @test all(φ2.sj .== s2)
    @test all(φ2.xk .== 0)
    @test φ2.l == l
    @test φ2.u == u
  end

  @testset "ShiftedNormL2InfBox with selected" begin
    sigma = 1.0
    omega = [1.0]
    n_group = 2
    l = [-1.0, -1.0, -1.0]
    u = [ 1.0,  1.0,  1.0]
    s = [0.0, 0.0, 0.0]
    x = zeros(3)
    selected = [1, 2]

    q   = [[3.0, 4.0, 0.0], [3.0, 4.0, 5.0], [3.0, 4.0, -5.0]]
    sol = [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0]]
    # Case 1 : q[3]=0  ∈ [l-s,u-s] → prox_zero(0,-1,1)  = 0
    # Case 2 : q[3]=5  > u-s=1     → prox_zero(5,-1,1)  = 1
    # Case 3 : q[3]=-5 < l-s=-1   → prox_zero(-5,-1,1) = -1

    for i = 1:3
      h = NormL2Inf(omega, n_group)
      ψ = shifted(h, x, l, u, selected)
      ω = shifted(ψ, s)
      y = similar(q[i])
      prox!(y, ω, q[i], sigma)
      @test isapprox(y, sol[i], atol = 1e-8)
    end

    h = NormL2Inf(omega, n_group)
    ψ = shifted(h, x, l, u, selected)
    ω = shifted(ψ, s)
    @test ω.selected == selected
  end
end