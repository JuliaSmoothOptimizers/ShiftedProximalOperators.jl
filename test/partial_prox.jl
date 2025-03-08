# test partial prox feature for operators that implement it
for op ∈ (:NormL0, :NormL1, :RootNormLhalf)
  @testset "shifted $op with box partial prox" begin
    λ = 3.14
    h = eval(op)(λ)
    n = 5
    l = zeros(n)
    u = ones(n)
    x = rand(n)
    s = rand(n)
    q = rand(n) .- 0.5
    Δ = 0.5
    χ = NormLinf(1.0)

    # compute the prox wrt all variables
    ψ = op == :RootNormLhalf ? shifted(h, x, Δ, χ) : shifted(h, x, l, u)
    if op == :RootNormLhalf
      # redefine l and u temporarily until this operator supports bound constraints
      l .= -Δ
      u .= Δ
    end
    ω = shifted(ψ, s)
    σ = 1.0
    y = prox(ω, q, σ)

    # compute a partial prox
    selected = 1:2:n
    ψ = op == :RootNormLhalf ? shifted(h, x, Δ, χ, selected) : shifted(h, x, l, u, selected)
    ω = shifted(ψ, s)
    σ = 1.0
    z = prox(ω, q, σ)
    p = min.(max.(q, l - s), u - s)

    for i = 1:n
      if i ∈ selected
        @test z[i] == y[i]
      else
        @test z[i] == p[i]
      end
    end

    # tests iprox with bounds
    if op == :NormL0 || op == :NormL1
      for d ∈ [ones(n), -ones(n), zeros(n)]
        y = iprox(ω, q, d)
        ω = shifted(ψ, s)
        z = iprox(ω, q, d)
        p = ShiftedProximalOperators.iprox_zero.(d, q, l - s, u - s)
        for i = 1:n
          if i ∈ selected
            @test z[i] == y[i]
          else
            @test z[i] == p[i]
          end
        end
      end
    end

    # tests iprox without bounds
    if op == :NormL0 || op == :NormL1
      ψ = shifted(h, x)
      # test iprox with d > 0
      for d ∈ [ones(n), 2 * ones(n)]
        y = iprox(ψ, q, d)
        σ = d[1]
        z = prox(ψ, q, σ)
        for i = 1:n
          if i ∈ selected
            @test z[i] == y[i]
          end
        end
      end
      # test iprox with d < 0
      for d ∈ [-ones(n), -2 * ones(n)]
        y = iprox(ψ, q, d)
        @test all(isinf.(y))
      end
      # test iprox with d = 0
      d = zeros(n)
      q1 = (λ + 1) * ones(n)
      y = iprox(ψ, q1, d)
      @test all(isinf.(y))
      q2 = zeros(n)
      y = iprox(ψ, q2, d)
      @test all(y .== -ψ.xk - ψ.sj)
    end
  end
end
