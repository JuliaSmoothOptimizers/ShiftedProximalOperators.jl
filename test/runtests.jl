using ProximalOperators
using ShiftedProximalOperators
using Test

# loop over operators without a trust region
for shifted_op ∈ (:ShiftedNormL0,)
  @testset "$shifted_op" begin
    ShiftedOp = eval(shifted_op)
    # test basic types and properties
    h = NormL0(1.2)
    x = ones(3)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}}
    @test all(ψ.x0 .== 0)
    @test all(ψ.x .== x)
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
    @test all(ψ.x0 .== 0)
    @test all(ψ.x .== y)

    # shift a shifted operator
    s = ones(3) / 2
    φ = shifted(ψ, s)
    @test all(φ.x0 .== x)
    @test all(φ.x .== s)
    @test φ(zeros(3)) == h(x + s)
    y = rand(3)
    @test φ(y) == h(x + s + y)

    # test different types
    h = NormL0(Float32(1.2))
    y = rand(Float32, 10)
    x = view(y, 1:2:10)
    ψ = shifted(h, x)
    @test typeof(ψ) == ShiftedOp{Float32, Vector{Float32}, SubArray{Float32, 1, Vector{Float32}, Tuple{StepRange{Int64, Int64}}, true}, Vector{Float32}}
    @test typeof(ψ.λ) == Float32
    @test ψ.λ == h.lambda
    @test ψ(zeros(Float32, 5)) == h(x)
  end
end

# loop over operators with a trust region
for shifted_op ∈ (:ShiftedNormL0BInf,)
  @testset "$shifted_op" begin
    ShiftedOp = eval(shifted_op)
    # test basic types and properties
    h = NormL0(1.2)
    x = ones(3)
    Δ = 0.5
    ψ = shifted(h, x, Δ)
    @test typeof(ψ) == ShiftedOp{Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}}
    @test all(ψ.x0 .== 0)
    @test all(ψ.x .== x)
    @test typeof(ψ.λ) == Float64
    @test ψ.λ == h.lambda
    @test ψ.Δ == Δ

    # test values
    @test ψ(zeros(3)) == h(x)
    y = rand(3)
    @test ψ(y) == h(x + y)

    # test prox
    # TODO

    # test shift update
    shift!(ψ, y)
    @test all(ψ.x0 .== 0)
    @test all(ψ.x .== y)

    # test radius update
    set_radius!(ψ, 1.1)
    @test ψ.Δ == 1.1

    # test different types
    h = NormL0(Float32(1.2))
    y = rand(Float32, 10)
    x = view(y, 1:2:10)
    ψ = shifted(h, x, Float32(0.5))
    @test typeof(ψ) == ShiftedOp{Float32, Vector{Float32}, SubArray{Float32, 1, Vector{Float32}, Tuple{StepRange{Int64, Int64}}, true}, Vector{Float32}}
    @test typeof(ψ.λ) == Float32
    @test ψ.λ == h.lambda
    @test ψ(zeros(Float32, 5)) == h(x)
  end
end
