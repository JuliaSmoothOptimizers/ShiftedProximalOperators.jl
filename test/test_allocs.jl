@testset "allocs" begin
  for op ∈ (:NormL0, :NormL1, :RootNormLhalf)
    h = eval(op)(1.0)
    n = 1000
    xk = rand(n)
    ψ = shifted(h, xk)
    y = rand(n)
    val = ψ(y)
    allocs = @allocated ψ(y)
    @test allocs == 16

    ψ = shifted(h, xk, -3.0, 4.0, rand(1:n, Int(n / 2)))
    val = ψ(y)
    allocs = @allocated ψ(y)
    @test allocs == 0
  end

  for op ∈ (:IndBallL0,)
    h = eval(op)(1)
    n = 1000
    xk = rand(n)
    ψ = shifted(h, xk)
    y = rand(n)
    val = ψ(y)
    allocs = @allocated ψ(y)
    @test allocs == 16

    χ = NormLinf(1.0)
    ψ = shifted(h, xk, 0.5, χ)
    val = ψ(y)
    allocs = @allocated ψ(y)
    @test allocs == 16
  end
end