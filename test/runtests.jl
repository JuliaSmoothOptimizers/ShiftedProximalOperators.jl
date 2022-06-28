using LinearAlgebra
using ProximalOperators
using ShiftedProximalOperators
using Test

VERSION ≥ v"1.7" && include("test_psvd.jl")

#test Created norms/standard proxes - TODO: come up with more robust test
for op ∈ (:RootNormLhalf,)
  @testset "$op" begin
    Op = eval(op)
    x = [0.4754 1.1741 0.1269 -0.6568]
    y = similar(x)
    q = [0.1097 1.1287 -0.29 1.2616]
    λ = 0.7788
    ν = 0.1056
    ytrue = [0.0 1.0893 -0.197463 1.22444]
    h = Op(λ)
    prox!(y, h, q, ν)
    @test sum((y - ytrue) .^ 2) ≤ 1e-11
  end
end
for op ∈ (:GroupNormL2,)
  @testset "$op" begin
    Op = eval(op)
    x = rand(6)
    y = similar(x)
    v = [1:3, collect(4:6)]
    yt = zeros(3)
    ytrue = similar(x)
    λ = rand(2)
    ν = rand()
    h = Op(λ, v)
    ysum = prox!(y, h, x, ν)
    ysumt = 0
    for i = 1:2
      ht = NormL2(λ[i])
      tt = prox!(yt, ht, x[v[i]], ν)
      ysumt += tt
      ytrue[v[i]] .= yt
    end
    @test sum((y - ytrue) .^ 2) ≤ 1e-11

    #test norm value
    lg2 = h(x)
    lg2_temp = 0
    for i = 1:2
      ht = NormL2(λ[i])
      lg2_temp += ht(x[v[i]])
    end
    @test abs(lg2 - lg2_temp) ≤ 1e-11
  end
end
# loop over operators without a trust region
for (op, shifted_op) ∈
    zip((:NormL0, :NormL1, :RootNormLhalf), (:ShiftedNormL0, :ShiftedNormL1, :ShiftedRootNormLhalf))
  @testset "$shifted_op" begin
    ShiftedOp = eval(shifted_op)
    Op = eval(op)
    # test basic types and properties
    h = Op(1.2)
    x = ones(3)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}}
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== x)
    @test typeof(ψ.λ) == Float64
    @test ψ.λ == h.lambda

    # test values
    @test ψ(zeros(3)) == h(x)
    y = rand(3)
    @test ψ(y) == h(x + y)

    # test prox
    # TODO

    # test shift update
    shift!(ψ, y)
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== y)

    # shift a shifted operator
    s = ones(3) / 2
    φ = shifted(ψ, s)
    @test all(φ.sj .== s)
    @test all(φ.xk .== x)
    @test φ(zeros(3)) == h(x + s)
    y = rand(3)
    @test φ(y) == h(x + s + y)

    # test different types
    h = Op(Float32(1.2))
    y = rand(Float32, 10)
    x = view(y, 1:2:10)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{
      Float32,
      SubArray{Float32, 1, Vector{Float32}, Tuple{StepRange{Int64, Int64}}, true},
      Vector{Float32},
      Vector{Float32},
    }
    @test typeof(ψ.λ) == Float32
    @test ψ.λ == h.lambda
    @test ψ(zeros(Float32, 5)) == h(x)
  end
end

# loop over integer operators without a trust region
for (op, shifted_op) ∈ zip((:IndBallL0,), (:ShiftedIndBallL0,))
  @testset "$shifted_op" begin
    ShiftedOp = eval(shifted_op)
    Op = eval(op)
    # test basic types and properties
    h = Op(1)
    x = ones(3)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{Int64, Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}}
    @test all(ψ.xk .== x)
    @test typeof(ψ.r) == Int64
    @test ψ.r == h.r

    # test values
    @test ψ(zeros(3)) == h(x)
    y = rand(3)
    @test ψ(y) == h(x + y)

    # test prox
    # TODO

    # test shift update
    shift!(ψ, y)
    @test all(ψ.xk .== y)

    # shift a shifted operator
    s = ones(3) / 2
    φ = shifted(ψ, s)
    @test all(φ.sj .== s)
    @test all(φ.xk .== x)
    @test φ(zeros(3)) == h(x + s)
    y = rand(3)
    @test φ(y) == h(x + s + y)

    # test different types
    h = Op(Int32(1))
    y = rand(Float32, 10)
    x = view(y, 1:2:10)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{
      Int32,
      Float32,
      SubArray{Float32, 1, Vector{Float32}, Tuple{StepRange{Int64, Int64}}, true},
      Vector{Float32},
      Vector{Float32},
    }
    @test typeof(ψ.r) == Int32
    @test ψ.r == h.r
    @test ψ(zeros(Int32, 5)) == h(x)
  end
end

# loop over operators with a trust region
for (op, tr, shifted_op) ∈ zip(
  (:NormL0, :NormL1, :NormL1, :RootNormLhalf),
  (:NormLinf, :NormLinf, :NormL2, :NormLinf),
  (:ShiftedNormL0BInf, :ShiftedNormL1BInf, :ShiftedNormL1B2, :ShiftedRootNormLhalfBinf),
)
  @testset "$shifted_op" begin
    ShiftedOp = eval(shifted_op)
    Op = eval(op)
    χ = eval(tr)(1.0)
    # test basic types and properties
    n = 5
    h = Op(1.0)
    x = ones(n)
    Δ = 0.01
    ψ = shifted(h, x, Δ, χ)
    @test typeof(ψ) == ShiftedOp{Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}}
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== x)
    @test typeof(ψ.λ) == Float64
    @test ψ.λ == h.lambda
    @test ψ.Δ == Δ

    # test values
    @test ψ(zeros(n)) == h(x)
    y = rand(n)
    y .*= ψ.Δ / ψ.χ(y) / 2
    @test ψ(y) == h(x + y)  # y inside the trust region
    @test ψ(3 * y) == Inf   # y outside the trust region

    # test prox
    ν = 1 / (9.1e+04)
    q =
      -ν .* [
        2631.441298528196,
        -533.9101219466443,
        466.56156501426733,
        1770.8953574224836,
        -2554.7769423950244,
      ]
    if "$shifted_op" == "ShiftedNormL0BInf"
      s_correct = [
        -0.010000000000000,
        0.005867144197216,
        -0.005127050164992,
        -0.010000000000000,
        0.010000000000000,
      ]
    elseif "$shifted_op" == "ShiftedNormL1B2"
      s_correct = [
        -0.006367076930786,
        0.001288947922799,
        -0.001130889587543,
        -0.004285677352167,
        0.006176811716709,
      ]
    elseif "$shifted_op" == "ShiftedNormL1BInf"
      s_correct = [
        -0.010000000000000,
        0.005856155186227,
        -0.005138039175981,
        -0.010000000000000,
        0.010000000000000,
      ]
    elseif "$shifted_op" == "ShiftedRootNormLhalfBinf"
      s_correct = [
        -0.010000000000000,
        0.005861665724748,
        -0.005132558825434,
        -0.010000000000000,
        0.010000000000000,
      ]
    end
    s = ShiftedProximalOperators.prox(ψ, q, ν)
    @test all(s .≈ s_correct)
    @test ψ.χ(s) ≤ ψ.Δ

    # test shift update
    shift!(ψ, y)
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== y)

    # test radius update
    set_radius!(ψ, 1.1)
    @test ψ.Δ == 1.1

    # shift a shifted operator
    s = ones(n)
    s /= 2 * ψ.χ(s)
    φ = shifted(ψ, s)
    @test all(φ.sj .== s)
    @test all(φ.xk .== x)
    @test φ(zeros(n)) == h(y + s)
    t = rand(n)
    t .*= ψ.Δ / ψ.χ(t) / 2
    @test φ(t) == h(y + s + t)  # y inside the trust region
    @test φ(3 * t) == Inf       # y outside the trust region

    # test different types
    h = Op(Float32(1.2))
    χ = eval(tr)(Float32(1.0))
    y = rand(Float32, 10)
    x = view(y, 1:2:10)
    ψ = shifted(h, x, Float32(0.5), χ)
    @test typeof(ψ) == ShiftedOp{
      Float32,
      SubArray{Float32, 1, Vector{Float32}, Tuple{StepRange{Int64, Int64}}, true},
      Vector{Float32},
      Vector{Float32},
    }
    @test typeof(ψ.λ) == Float32
    @test ψ.λ == h.lambda
    @test ψ(zeros(Float32, 5)) == h(x)
  end
end

# loop over operators with a trust region
for (op, tr, shifted_op) ∈ zip((:IndBallL0,), (:NormLinf,), (:ShiftedIndBallL0BInf,))
  @testset "$shifted_op" begin
    ShiftedOp = eval(shifted_op)
    χ = eval(tr)(1.0)
    Op = eval(op)
    # test basic types and properties
    h = Op(1)
    x = ones(3)
    Δ = 0.5
    ψ = shifted(h, x, Δ, χ)
    @test typeof(ψ) == ShiftedOp{Int64, Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}}
    @test all(ψ.xk .== x)
    @test typeof(ψ.r) == Int64
    @test ψ.r == h.r
    @test ψ.Δ == Δ

    # test values
    @test ψ(zeros(3)) == h(x)
    y = rand(3)
    y .*= ψ.Δ / ψ.χ(y) / 2
    @test ψ(y) == h(x + y)  # y inside the trust region
    @test ψ(3 * y) == Inf   # y outside the trust region

    # test prox
    # TODO

    # test shift update
    shift!(ψ, y)
    @test all(ψ.xk .== y)

    # test radius update
    set_radius!(ψ, 1.1)
    @test ψ.Δ == 1.1

    # shift a shifted operator
    s = ones(3) / 2
    φ = shifted(ψ, s)
    @test all(φ.sj .== s)
    @test all(φ.xk .== x)
    @test φ(zeros(3)) == h(x + s)
    y = rand(3)
    y .*= ψ.Δ / ψ.χ(y) / 2
    @test φ(y) == h(x + s + y)  # y inside the trust region
    @test φ(3 * y) == Inf       # y outside the trust region

    # test different types
    h = Op(Int32(1))
    χ = eval(tr)(Float32(1.0))
    y = rand(Float32, 10)
    x = view(y, 1:2:10)
    ψ = shifted(h, x, Float32(0.5), χ)
    @test typeof(ψ) == ShiftedOp{
      Int32,
      Float32,
      SubArray{Float32, 1, Vector{Float32}, Tuple{StepRange{Int64, Int64}}, true},
      Vector{Float32},
      Vector{Float32},
    }
    @test typeof(ψ.r) == Int32
    @test ψ.r == h.r
    @test ψ(zeros(Float32, 5)) == h(x)
  end
end

# loop over separable convex operators with separable trust region
for (op, tr) ∈ zip((:NormL1,), (:NormLinf,))
  @testset "prox for $op with $tr trust region" begin
    χ = eval(tr)(1.0)
    h = eval(op)(1.0)
    n = 4
    Δ = 2 * rand()
    q = 2 * (rand(n) .- 0.5)
    ν = rand()

    # shift once
    xk = rand(n) .- 0.5
    ψ = shifted(h, xk, Δ, χ)

    # check prox
    p1 = ProximalOperators.prox(h, xk + q, ν)[1]
    p1 .= min.(max.(p1, xk .- Δ), xk .+ Δ) - xk
    p2 = ShiftedProximalOperators.prox(ψ, q, ν)
    @test all(p1 .≈ p2)

    # shift a second time
    sj = rand(n) .- 0.5
    ω = shifted(ψ, sj)

    # check prox
    p1 = ProximalOperators.prox(h, xk + sj + q, ν)[1]
    p1 .= min.(max.(p1, xk .- Δ), xk .+ Δ) - (xk + sj)
    p2 = ShiftedProximalOperators.prox(ω, q, ν)
    @test all(p1 .≈ p2)
  end
end
