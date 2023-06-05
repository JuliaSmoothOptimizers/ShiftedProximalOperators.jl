for op ∈ (:NormL0, :NormL1, :RootNormLhalf)
  h = eval(op)(1.0)
  n = 1000
  xk = rand(n)
  ψ = shifted(h, xk, -3.0, 4.0, rand(1:n, Int(n / 2)))
  y = rand(n)
  val = ψ(y)
  allocs = @allocated ψ(y)
  @test allocs == 0
end