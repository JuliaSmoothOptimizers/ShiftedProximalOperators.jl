"""
    @wrappedallocs(expr)

Given an expression, this macro wraps that expression inside a new function
which will evaluate that expression and measure the amount of memory allocated
by the expression. Wrapping the expression in a new function allows for more
accurate memory allocation detection when using global variables (e.g. when
at the REPL).

This code is based on that of https://github.com/JuliaAlgebra/TypedPolynomials.jl/blob/master/test/runtests.jl

For example, `@wrappedallocs(x + y)` produces:

```julia
function g(x1, x2)
    @allocated x1 + x2
end
g(x, y)
```

You can use this macro in a unit test to verify that a function does not
allocate:

```
@test @wrappedallocs(x + y) == 0
```
"""
macro wrappedallocs(expr)
  argnames = [gensym() for a in expr.args]
  quote
    function g($(argnames...))
      @allocated $(Expr(expr.head, argnames...))
    end
    $(Expr(:call, :g, [esc(a) for a in expr.args]...))
  end
end

@testset "allocs" begin

  for (op, composite_op) ∈ ((:NormL2, :CompositeNormL2),)
    CompositeOp = eval(composite_op)

    function c!(z,x)
      z[1] = 2*x[1] - x[4]
      z[2] = x[2] + x[3] 
    end
    function J!(z,x)
      z.vals .= Float64[2.0,1.0,1.0,-1.0]
    end
    λ = 3.62
    Op = eval(op)
    h = Op(λ)

    b = zeros(Float64,2)
    A = SparseMatrixCOO(Float64[2 0 0 -1;0 1 1 0])

    ψ = CompositeOp(λ,c!,J!,A,b)

    # test shifted operator
    xk = [0.0,1.1741,0.0,-0.4754]
    ϕ = shifted(ψ,xk)

    # test prox 
    x = [0.1097,1.1287,-0.29,1.2616]
    y = similar(x)
    ν = 0.1056
    @test @wrappedallocs(prox!(y,ϕ,x,ν)) == 0
  end
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
    @test allocs == 16
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

  for op ∈ (:NormL0, :NormL1)
    h = eval(op)(1.0)
    n = 1000
    xk = rand(n)
    ψ = shifted(h, xk)
    y = rand(n)
    d = rand(n)
    @test @wrappedallocs(prox!(y, ψ, y, 1.0)) == 0
    @test @wrappedallocs(iprox!(y, ψ, y, d)) == 0

    ψ = shifted(h, xk, -3.0, 4.0, rand(1:n, Int(n / 2)))
    @test @wrappedallocs(prox!(y, ψ, y, 1.0)) == 0
    @test @wrappedallocs(iprox!(y, ψ, y, d)) == 0
  end

  for (op, shifted_op) ∈ zip((:Rank, :Nuclearnorm), (:ShiftedRank, :ShiftedNuclearnorm))
    ShiftedOp = eval(shifted_op)
    Op = eval(op)
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
    @test @wrappedallocs(prox!(y, h, x, γ)) == 0
    @test @wrappedallocs(prox!(y, f, q, γ)) == 0
  end
end
