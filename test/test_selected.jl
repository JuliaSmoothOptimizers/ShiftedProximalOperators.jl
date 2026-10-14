# shifted(h, xk, selected): h only applies to the selected variables.
function test_selected(h, n, selected)
  unselected = setdiff(1:n, selected)
  x = rand(n)
  s = rand(n)
  q = rand(n) .- 0.5
  σ = 0.7

  ψ = shifted(h, x, selected)
  ψref = shifted(h, x[selected])
  @test ψ.h === h
  @test ψ.xk == x[selected]

  y = copy(prox(ψ, q, σ))
  @test y[selected] ≈ prox(ψref, q[selected], σ)
  @test y[unselected] == q[unselected]
  @test ψ(y) ≈ h(x[selected] + y[selected])

  # shifted twice
  ω = shifted(ψ, s)
  ωref = shifted(ψref, s[selected])
  z = copy(prox(ω, q, σ))
  @test z[selected] ≈ prox(ωref, q[selected], σ)
  @test z[unselected] == q[unselected]
  @test ω(z) ≈ h(x[selected] + s[selected] + z[selected])

  # shift! updates xk, which ψ refers to
  x2 = rand(n)
  shift!(ψ, x2)
  @test x[selected] == x2[selected]
  @test ψ(y) ≈ h(x2[selected] + y[selected])

  @test_throws ArgumentError set_radius!(ψ, 1.0)
  @test_throws ArgumentError set_bounds!(ψ, -1.0, 1.0)
  return ψ
end

@testset "shifted with selected variables" begin
  for h ∈ (NormL0(0.3), NormL1(0.3), RootNormLhalf(0.3), IndBallL0(2))
    @testset "$(typeof(h).name.name)" begin
      test_selected(h, 6, [1, 2, 4])
      test_selected(h, 6, 2:5)
    end
  end

  # a non-separable regularizer, on a 2 × 2 matrix made of 4 of the 6 variables
  @testset "Nuclearnorm" begin
    F = psvd_workspace_dd(zeros(2, 2), full = false)
    test_selected(Nuclearnorm(0.3, ones(2, 2), F), 6, [1, 3, 4, 6])
  end

  # agrees with the box variant with infinite bounds
  @testset "agrees with ShiftedNormL1Box" begin
    n, selected = 6, [1, 2, 4]
    x, q = rand(n), rand(n) .- 0.5
    y = copy(prox(shifted(NormL1(0.3), x, selected), q, 0.7))
    @test y ≈ prox(shifted(NormL1(0.3), x, -Inf, Inf, selected), q, 0.7)
  end

  @testset "iprox" begin
    n, selected = 6, [1, 2, 4]
    unselected = setdiff(1:n, selected)
    x, g = rand(n), rand(n) .- 0.5
    d = 2 * ones(n)
    for h ∈ (NormL0(0.3), NormL1(0.3))
      ψ = shifted(h, x, selected)
      y = copy(iprox(ψ, g, d))
      @test y[selected] ≈ iprox(shifted(h, x[selected]), g[selected], d[selected])
      @test y[unselected] ≈ -g[unselected] ./ d[unselected]
      @test_throws AssertionError iprox(ψ, g, zeros(n))
    end
  end

  # selecting all the variables returns the plain shifted regularizer
  @testset "all variables selected" begin
    n = 6
    x = rand(n)
    for h ∈ (NormL0(0.3), NormL1(0.3), RootNormLhalf(0.3), IndBallL0(2))
      for selected ∈ (1:n, collect(1:n))
        ψ = shifted(h, x, selected)
        @test typeof(ψ) == typeof(shifted(h, x))
        @test ψ.xk === x
      end
      # a permutation of all the variables is not a no-op in general
      @test shifted(h, x, [2, 1, 3, 4, 5, 6]) isa ShiftedSelected
    end

    # a shifted regularizer passed as h, as in the subsolvers of trust-region methods
    ψtr = shifted(NormL1(0.3), x, 1.0, NormLinf(1.0))
    ω = shifted(ψtr, rand(n), 1:n)
    @test typeof(ω) == typeof(ψtr)
    set_radius!(ω, 0.5)
    @test ω.u == 0.5

    # trust regions without a box variant
    χ2, χinf = NormL2(1.0), NormLinf(1.0)
    G = GroupNormL2([0.3, 0.3], [[1, 2, 3], [4, 5, 6]])
    for (h, χ) ∈ ((NormL1(0.3), χ2), (NormL2(0.3), χinf), (G, χinf), (IndBallL0(2), χinf))
      @test typeof(shifted(h, x, 1.0, χ, 1:n)) == typeof(shifted(h, x, 1.0, χ))
      @test_throws ArgumentError shifted(h, x, 1.0, χ, 1:(n - 1))
    end

    # trust regions with a box variant keep honoring selected
    for h ∈ (NormL0(0.3), NormL1(0.3), RootNormLhalf(0.3))
      @test typeof(shifted(h, x, 1.0, χinf, 1:n)) == typeof(shifted(h, x, 1.0, χinf))
      @test shifted(h, x, 1.0, χinf, [1, 2, 4]).selected == [1, 2, 4]
    end
  end

  # shifting twice reuses the selection
  ψ = shifted(NormL1(0.3), rand(6), [1, 2, 4])
  ω = shifted(ψ, rand(6))
  @test ω.selected === ψ.selected
  @test ω.unselected === ψ.unselected

  @test_throws BoundsError shifted(NormL1(1.0), rand(3), [1, 4])
  @test_throws ArgumentError shifted(NormL1(1.0), rand(3), [1, 1])
end
