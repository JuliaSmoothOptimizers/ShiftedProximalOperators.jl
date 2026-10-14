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

  @test_throws BoundsError shifted(NormL1(1.0), rand(3), [1, 4])
  @test_throws ArgumentError shifted(NormL1(1.0), rand(3), [1, 1])
end
