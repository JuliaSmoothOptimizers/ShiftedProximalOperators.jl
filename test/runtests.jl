using LinearAlgebra
using ProximalOperators
using ShiftedProximalOperators
using Test

include("test_psvd.jl")

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
# Test NormL2 separately
for (op, shifted_op) ∈ zip((:NormL2,), (:ShiftedGroupNormL2,))
  @testset "$shifted_op" begin
    ShiftedOp = eval(shifted_op)
    Op = eval(op)
    λ = rand()
    # test basic types and properties
    h = Op(λ)
    x = ones(6)
    ν = rand()
    q = randn(size(x))
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{
      Float64,
      Vector{Float64},
      Vector{Colon},
      Vector{Float64},
      Vector{Float64},
      Vector{Float64},
    }
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== x)
    @test typeof(ψ.λ) == Vector{Float64}
    @test sum(ψ.λ .== h.lambda) == length(h.lambda)

    # test values
    @test ψ(zeros(6)) .== h(x)
    yψ = similar(x)
    yp = similar(x)
    y = rand(6)
    @test ψ(y) == h(x + y)

    # test prox
    prox!(yψ, ψ, q, ν)
    # test prox
    prox!(yψ, ψ, q, ν)
    idx = ψ.h.idx
    ht = Op(λ)
    prox!(yp, ht, q + x, ν)
    @test sqrt(sum((yψ - (yp - x)) .^ 2)) ≤ 1e-11

    # test shift update
    shift!(ψ, y)
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== y)

    # shift a shifted operator
    s = ones(6) / 2
    φ = shifted(ψ, s)
    @test all(φ.sj .== s)
    @test all(φ.xk .== x)
    @test φ(zeros(6)) == h(x + s)
    y = rand(6)
    @test φ(y) == h(x + s + y)

    # test different types
    h = Op(Float32(1.2))
    y = rand(Float32, 10)
    x = view(y, 1:2:10)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{
      Float32,
      Vector{Float32},
      Vector{Colon},
      SubArray{Float32, 1, Vector{Float32}, Tuple{StepRange{Int64, Int64}}, true},
      Vector{Float32},
      Vector{Float32},
    }
    @test typeof(ψ.λ) == Vector{Float32}
    @test ψ.λ == [h.lambda]
    @test ψ(zeros(Float32, 5)) == h(x)
  end
end

for (op, shifted_op) ∈ zip((:GroupNormL2,), (:ShiftedGroupNormL2,))
  @testset "$shifted_op" begin
    ShiftedOp = eval(shifted_op)
    Op = eval(op)
    v = [collect(1:3), collect(4:6)]
    λ = rand(2)
    # test basic types and properties
    h = Op(λ, v)
    x = ones(6)
    ν = rand()
    q = randn(size(x))
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{
      Float64,
      Vector{Float64},
      Vector{Vector{Int64}},
      Vector{Float64},
      Vector{Float64},
      Vector{Float64},
    }
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== x)
    @test typeof(ψ.λ) == Vector{Float64}
    @test sum(ψ.λ .== h.lambda) == length(h.lambda)

    # test values
    @test ψ(zeros(6)) .== h(x)
    yψ = similar(x)
    yp = similar(x)
    y = rand(6)
    @test ψ(y) == h(x + y)

    # test prox
    prox!(yψ, ψ, q, ν)
    # test prox
    prox!(yψ, ψ, q, ν)
    idx = ψ.h.idx
    for i = 1:length(λ)
      ht = NormL2(λ[i])
      ytemp = zeros(size(idx[i]))
      prox!(ytemp, ht, q[idx[i]] + x[idx[i]], ν)
      yp[idx[i]] .= ytemp
    end
    @test sqrt(sum((yψ - (yp - x)) .^ 2)) ≤ 1e-11

    # test shift update
    shift!(ψ, y)
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== y)

    # shift a shifted operator
    s = ones(6) / 2
    φ = shifted(ψ, s)
    @test all(φ.sj .== s)
    @test all(φ.xk .== x)
    @test φ(zeros(6)) == h(x + s)
    y = rand(6)
    @test φ(y) == h(x + s + y)

    # test different types
    h = Op([Float32(1.2)])
    y = rand(Float32, 10)
    x = view(y, 1:2:10)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{
      Float32,
      Vector{Float32},
      Vector{Colon},
      SubArray{Float32, 1, Vector{Float32}, Tuple{StepRange{Int64, Int64}}, true},
      Vector{Float32},
      Vector{Float32},
    }
    @test typeof(ψ.λ) == Vector{Float32}
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
for (op, tr, shifted_op) ∈ zip((:NormL2,), (:NormLinf,), (:ShiftedGroupNormL2Binf,))
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
    @test typeof(ψ) == ShiftedOp{
      Float64,
      Vector{Float64},
      Vector{Colon},
      Vector{Float64},
      Vector{Float64},
      Vector{Float64},
    }
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== x)
    @test typeof(ψ.λ) == Vector{Float64}
    @test ψ.λ == [h.lambda]
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
    s_correct = [
      -0.010000000000000,
      0.005862191941930,
      -0.005131948291800,
      -0.010000000000000,
      0.010000000000000,
    ]
    s = ShiftedProximalOperators.prox(ψ, q, ν)
    @test all(s .≈ s_correct)
    @test ψ.χ(s) ≤ ψ.Δ || ψ.χ(s) ≈ ψ.Δ

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
      Vector{Float32},
      Vector{Colon},
      SubArray{Float32, 1, Vector{Float32}, Tuple{StepRange{Int64, Int64}}, true},
      Vector{Float32},
      Vector{Float32},
    }
    @test typeof(ψ.λ) == Vector{Float32}
    @test ψ.λ == [h.lambda]
    @test ψ(zeros(Float32, 5)) == h(x)
  end
end
for (op, tr, shifted_op) ∈ zip((:GroupNormL2,), (:NormLinf,), (:ShiftedGroupNormL2Binf,))
  @testset "$shifted_op" begin
    ShiftedOp = eval(shifted_op)
    Op = eval(op)
    χ = eval(tr)(1.0)
    # test basic types and properties
    n = 6
    x = ones(n)
    Δ = 0.01
    v = [collect(1:3), collect(4:6)]
    λ = [0.396767474230670, 0.538816734003357]

    h = Op(λ, v)
    x = ones(6)
    ν = 0.419194514403295
    q = [
      -0.649013765191241,
      1.181166041965532,
      -0.758453297283692,
      -1.109613038501522,
      -0.845551240007797,
      -0.572664866457950,
    ]
    ψ = shifted(h, x, Δ, χ)
    @test typeof(ψ) == ShiftedOp{
      Float64,
      Vector{Float64},
      Vector{Vector{Int64}},
      Vector{Float64},
      Vector{Float64},
      Vector{Float64},
    }
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== x)
    @test typeof(ψ.λ) == Vector{Float64}
    @test sum(ψ.λ .== h.lambda) == length(h.lambda)

    # test values
    @test ψ(zeros(6)) .== h(x)
    y = rand(6)
    y .*= ψ.Δ / ψ.χ(y) / 2
    @test ψ(y) == h(x + y)  # y inside the trust region
    @test ψ(3 * y) == Inf   # y outside the trust region
    yψ = similar(x)
    yp = similar(x)

    # test prox
    s_correct = [
      -0.010000000000000,
      0.010000000000000,
      -0.010000000000000,
      -0.010000000000000,
      -0.010000000000000,
      -0.010000000000000,
    ]
    s = ShiftedProximalOperators.prox(ψ, q, ν)
    @test all(s .≈ s_correct)
    @test ψ.χ(s) ≤ ψ.Δ || ψ.χ(s) ≈ ψ.Δ

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
    h = Op([Float32(1.2)])
    χ = eval(tr)(Float32(1.0))
    y = rand(Float32, 10)
    x = view(y, 1:2:10)
    ψ = shifted(h, x, Float32(0.5), χ)
    @test typeof(ψ) == ShiftedOp{
      Float32,
      Vector{Float32},
      Vector{Colon},
      SubArray{Float32, 1, Vector{Float32}, Tuple{StepRange{Int64, Int64}}, true},
      Vector{Float32},
      Vector{Float32},
    }
    @test typeof(ψ.λ) == Vector{Float32}
    @test ψ.λ == h.lambda
    @test ψ(zeros(Float32, 5)) == h(x) # throws error because IndBallLinf(ψ.Δ)(ψ.sj + zeros(x)) == 0.0f0 -> julia does not want to add it to Float32[...]
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

# loop over Rank function

for (op, shifted_op) ∈ zip((:Rank,), (:ShiftedRank,))
  @testset "$shifted_op" begin
    ShiftedOp = eval(shifted_op)
    Op = eval(op)
    # test basic types and properties
    F = psvd_workspace_dd(zeros(2, 2), full = false)
    h = Op(1.0, ones(2, 2), F)
    x = ones(4)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{
      Float64,
      Matrix{Float64},
      Float64,
      Float64,
      Matrix{Float64},
      Vector{Float64},
      Vector{Float64},
      Vector{Float64},
    }
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== x)
    @test typeof(ψ.λ) == Float64
    @test ψ.λ == h.lambda

    # test basic types and properties for the new constructor
    h = Op(1.0, ones(2, 2))
    x = ones(4)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{
      Float64,
      Matrix{Float64},
      Float64,
      Float64,
      Matrix{Float64},
      Vector{Float64},
      Vector{Float64},
      Vector{Float64},
    }
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== x)
    @test typeof(ψ.λ) == Float64
    @test ψ.λ == h.lambda

    # test values
    @test ψ(zeros(4)) == h(x)
    y = rand(4)
    @test ψ(y) == h(x + y)

    # test prox
    # TODO

    # test shift update
    shift!(ψ, y)
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== y)

    # shift a shifted operator
    s = ones(4) / 2
    φ = shifted(ψ, s)
    @test all(φ.sj .== s)
    @test all(φ.xk .== x)
    @test φ(zeros(4)) == h(x + s)
    y = rand(4)
    @test φ(y) == h(x + s + y)

    # test different types
    F = psvd_workspace_dd(zeros(2, 2), full = false)
    h = Op(Float32(1.2), ones(2, 2), F)
    y = rand(Float32, 8)
    x = view(y, 1:2:8)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{
      Float32,
      Matrix{Float64},
      Float64,
      Float64,
      Matrix{Float64},
      SubArray{Float32, 1, Vector{Float32}, Tuple{StepRange{Int64, Int64}}, true},
      Vector{Float32},
      Vector{Float32},
    }
    @test typeof(ψ.λ) == Float32
    @test ψ.λ == h.lambda
    @test ψ(zeros(Float32, 4)) == h(x)

    # test more sophisticated examples
    # Diagonal Matrix (n,n)
    n = 10
    λ = 10.0
    st1 = rand(n)
    x = vec(reshape(Diagonal(st1), n^2, 1))
    q = x .^ 2
    s = x / 2
    F = psvd_workspace_dd(zeros(n, n), full = false)
    h = Op(λ, ones(n, n), F)
    f = shifted(shifted(h, x), s)
    y = zeros(n^2)
    k = NormL0(λ)
    t = ProximalOperators.prox(k, st1 + st1 .^ 2 + st1 / 2, λ)[1]
    @test all(Diagonal(t - st1 - st1 / 2) .≈ reshape(prox!(y, f, q, λ), n, n))

    # Rectangular Matrix (m,n)
    m = 10
    n = 11
    λ = 1.0
    γ = 5.0
    x = vec(reshape(rand(m, n), m * n, 1))
    q = vec(reshape(rand(m, n), m * n, 1))
    s = vec(reshape(rand(m, n), m * n, 1))
    F = psvd_workspace_dd(zeros(m, n), full = false)
    h = Op(λ, ones(m, n), F)
    f = ShiftedOp(h, x, s, true)
    y = zeros(m * n)
    k = NormL0(λ)
    Q = svd(reshape(q + s + x, m, n))
    t = ProximalOperators.prox(k, Q.S, γ)[1]
    @test all(Q.U * Diagonal(t) * Q.Vt - reshape(x + s, m, n) .≈ reshape(prox!(y, f, q, γ), m, n))
  end
end

for (op, shifted_op) ∈ zip((:Cappedl1,), (:ShiftedCappedl1,))
  @testset "$shifted_op" begin
    ShiftedOp = eval(shifted_op)
    Op = eval(op)
    # test basic types and properties
    θ = 1.0
    F = psvd_workspace_dd(zeros(2, 2), full = false)
    h = Op(1.0, θ, ones(2, 2), F)
    x = ones(4)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{
      Float64,
      Matrix{Float64},
      Float64,
      Float64,
      Matrix{Float64},
      Vector{Float64},
      Vector{Float64},
      Vector{Float64},
    }
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== x)
    @test typeof(ψ.λ) == Float64
    @test ψ.λ == h.lambda

    # test basic types and properties for the new constructor
    h = Op(1.0, θ, ones(2, 2))
    x = ones(4)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{
      Float64,
      Matrix{Float64},
      Float64,
      Float64,
      Matrix{Float64},
      Vector{Float64},
      Vector{Float64},
      Vector{Float64},
    }
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== x)
    @test typeof(ψ.λ) == Float64
    @test ψ.λ == h.lambda

    # test values
    @test ψ(zeros(4)) == h(x)
    y = rand(4)
    @test ψ(y) == h(x + y)

    # test prox
    # TODO

    # test shift update
    shift!(ψ, y)
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== y)

    # shift a shifted operator
    s = ones(4) / 2
    φ = shifted(ψ, s)
    @test all(φ.sj .== s)
    @test all(φ.xk .== x)
    @test φ(zeros(4)) == h(x + s)
    y = rand(4)
    @test φ(y) == h(x + s + y)

    # test different types
    F = psvd_workspace_dd(zeros(2, 2), full = false)
    h = Op(Float32(1.2), Float32(θ), ones(2, 2), F)
    y = rand(Float32, 8)
    x = view(y, 1:2:8)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{
      Float32,
      Matrix{Float64},
      Float64,
      Float64,
      Matrix{Float64},
      SubArray{Float32, 1, Vector{Float32}, Tuple{StepRange{Int64, Int64}}, true},
      Vector{Float32},
      Vector{Float32},
    }
    @test typeof(ψ.λ) == Float32
    @test ψ.λ == h.lambda
    @test ψ(zeros(Float32, 4)) == h(x)

    # test more sophisticated examples
    # Diagonal Matrix (n,n)
    n = 10
    θ = 1.0
    λ = 10.0
    st1 = rand(n)
    x = vec(reshape(Diagonal(st1), n^2, 1))
    q = x .^ 2
    s = x / 2
    F = psvd_workspace_dd(zeros(n, n), full = false)
    h = Op(λ, θ, ones(n, n), F)
    f = shifted(shifted(h, x), s)
    y = zeros(n^2)
    k = NormL0(λ)
    t = ProximalOperators.prox(k, st1 + st1 .^ 2 + st1 / 2, λ)[1]
    @test all(Diagonal(t - st1 - st1 / 2) .≈ reshape(prox!(y, f, q, λ), n, n))
  end
end

for (op, shifted_op) ∈ zip((:Nuclearnorm,), (:ShiftedNuclearnorm,))
  @testset "$shifted_op" begin
    ShiftedOp = eval(shifted_op)
    Op = eval(op)
    # test basic types and properties
    F = psvd_workspace_dd(zeros(2, 2), full = false)
    h = Op(1.0, ones(2, 2), F)
    x = ones(4)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{
      Float64,
      Matrix{Float64},
      Float64,
      Float64,
      Matrix{Float64},
      Vector{Float64},
      Vector{Float64},
      Vector{Float64},
    }
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== x)
    @test typeof(ψ.λ) == Float64
    @test ψ.λ == h.lambda

    # test basic types and properties for the new constructor
    h = Op(1.0, ones(2, 2))
    x = ones(4)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{
      Float64,
      Matrix{Float64},
      Float64,
      Float64,
      Matrix{Float64},
      Vector{Float64},
      Vector{Float64},
      Vector{Float64},
    }
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== x)
    @test typeof(ψ.λ) == Float64
    @test ψ.λ == h.lambda

    # test values
    @test ψ(zeros(4)) == h(x)
    y = rand(4)
    @test ψ(y) == h(x + y)

    # test prox
    # TODO

    # test shift update
    shift!(ψ, y)
    @test all(ψ.sj .== 0)
    @test all(ψ.xk .== y)

    # shift a shifted operator
    s = ones(4) / 2
    φ = shifted(ψ, s)
    @test all(φ.sj .== s)
    @test all(φ.xk .== x)
    @test φ(zeros(4)) == h(x + s)
    y = rand(4)
    @test φ(y) == h(x + s + y)

    # test different types
    F = psvd_workspace_dd(zeros(2, 2), full = false)
    h = Op(Float32(1.2), ones(2, 2), F)
    y = rand(Float32, 8)
    x = view(y, 1:2:8)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{
      Float32,
      Matrix{Float64},
      Float64,
      Float64,
      Matrix{Float64},
      SubArray{Float32, 1, Vector{Float32}, Tuple{StepRange{Int64, Int64}}, true},
      Vector{Float32},
      Vector{Float32},
    }
    @test typeof(ψ.λ) == Float32
    @test ψ.λ == h.lambda
    @test ψ(zeros(Float32, 4)) == h(x)

    # test more sophisticated examples
    # Diagonal Matrix (n,n)
    n = 10
    λ = 10.0
    st1 = rand(n)
    x = vec(reshape(Diagonal(st1), n^2, 1))
    q = x .^ 2
    s = x / 2
    F = psvd_workspace_dd(zeros(n, n), full = false)
    h = Op(λ, ones(n, n), F)
    f = shifted(shifted(h, x), s)
    y = zeros(n^2)
    k = NormL1(λ)
    t = ProximalOperators.prox(k, st1 + st1 .^ 2 + st1 / 2, λ)[1]
    @test all(Diagonal(t - st1 - st1 / 2) .≈ reshape(prox!(y, f, q, λ), n, n))

    # Rectangular Matrix (m,n)
    m = 10
    n = 11
    λ = 1.0
    γ = 5.0
    x = vec(reshape(rand(m, n), m * n, 1))
    q = vec(reshape(rand(m, n), m * n, 1))
    s = vec(reshape(rand(m, n), m * n, 1))
    F = psvd_workspace_dd(zeros(m, n), full = false)
    h = Op(λ, ones(m, n), F)
    f = ShiftedOp(h, x, s, true)
    y = zeros(m * n)
    k = NormL1(λ)
    Q = svd(reshape(q + s + x, m, n))
    t = ProximalOperators.prox(k, Q.S, γ)[1]
    @test all(Q.U * Diagonal(t) * Q.Vt - reshape(x + s, m, n) .≈ reshape(prox!(y, f, q, γ), m, n))
  end
end
