export ShiftedCompositeNormL1

@doc raw"""
    ShiftedCompositeNormL1(h, c!, J!, A, b)

Returns the shift of a function c composed with the ``\ell_{1}`` norm (see CompositeNormL1.jl).
Here, c is linearized i.e, ``c(x+s) \approx c(x) + J(x)s``. 
```math
f(s) = λ \|c(x) + J(x)s\|_1
```
where ``\lambda > 0``. c! and J! should be functions
```math
\begin{aligned}
&c(x) : \mathbb{R}^n \xrightarrow[]{} \mathbb{R}^m \\
&J(x) : \mathbb{R}^n \xrightarrow[]{} \mathbb{R}^{m\times n}
\end{aligned}
```
such that J is the Jacobian of c. A and b should respectively be a matrix and a vector which can respectively store the values of J and c.
"""
mutable struct ShiftedCompositeNormL1{
  R <: Real,
  V0 <: Function,
  V1 <: Function,
  V2 <: AbstractMatrix{R},
  V3 <: AbstractVector{R},
} <: ShiftedCompositeProximableFunction
  h::NormL1{R}
  c!::V0
  J!::V1
  A::V2
  b::V3
  res::V3
  sol::V3
  dsol::V3
  function ShiftedCompositeNormL1(
    h::NormL1{R},
    c!::Function,
    J!::Function,
    A::AbstractMatrix{R},
    b::AbstractVector{R},
  ) where {R <: Real}
    res = similar(b)
    sol = similar(b)
    dsol = similar(b)
    if length(b) != size(A,1)
      error("ShiftedCompositeNormL1: Wrong input dimensions, there should be as many constraints as rows in the Jacobian")
    end
    new{R,typeof(c!),typeof(J!),typeof(A),typeof(b)}(h,c!,J!,A,b,res,sol,dsol)
  end
end


shifted(h::NormL1{R}, c!::Function,J!::Function,A::AbstractMatrix{R},b::AbstractVector{R}, xk :: AbstractVector{R}) where {R <: Real} = begin
  c!(b,xk)
  J!(A,xk)
  ShiftedCompositeNormL1(h,c!,J!,A,b)
end

shifted(
  ψ::ShiftedCompositeNormL1{R, V0, V1, V2, V3},
  xk::AbstractVector{R},
) where {R <: Real, V0 <: Function, V1 <: Function, V2 <: AbstractMatrix{R},V3<: AbstractVector{R}} = begin
  b = similar(ψ.b)
  ψ.c!(b,xk)
  A = similar(ψ.A)
  ψ.J!(A,xk)
  ShiftedCompositeNormL1(ψ.h, ψ.c!,ψ.J!,A,b)
end
 
shifted(
  ψ::CompositeNormL1{R,V0,V1,V2,V3},
  xk::AbstractVector{R}
) where {R <: Real, V0 <: Function, V1 <: Function, V2 <: AbstractMatrix{R}, V3 <: AbstractVector{R}} = begin
  b = similar(ψ.b)
  ψ.c!(b,xk)
  A = similar(ψ.A)
  ψ.J!(A,xk)
  ShiftedCompositeNormL1(ψ.h,ψ.c!,ψ.J!,A,b)
end

fun_name(ψ::ShiftedCompositeNormL1) = "shifted L1 norm"
fun_expr(ψ::ShiftedCompositeNormL1) = "t ↦ ‖c(xk) + J(xk)t‖₁"
fun_params(ψ::ShiftedCompositeNormL1) = "c(xk) = $(ψ.b)\n" * " "^14 * "J(xk) = $(ψ.A)\n"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedCompositeNormL1{R, V0, V1, V2, V3},
  q::AbstractVector{R},
  σ::R
) where {R <: Real, V0 <: Function,V1 <:Function,V2 <: AbstractMatrix{R}, V3 <: AbstractVector{R}}

  mul!(ψ.res, ψ.A, q)
  ψ.res .+= ψ.b

  spmat = qrm_spmat_init(ψ.A; sym=false)
  spfct = qrm_spfct_init(spmat)
  qrm_analyse!(spmat, spfct; transp='t')
  qrm_set(spfct, "qrm_keeph", 0)
  qrm_factorize!(spmat, spfct, transp='t')

  qrm_solve!(spfct, ψ.res, y, transp='t')
  qrm_solve!(spfct, y, ψ.sol, transp='n')

  # 1 step of iterative refinement
  mul!(y, ψ.A', ψ.sol)
  mul!(ψ.dsol, ψ.A, y)

  ψ.res .-= ψ.dsol

  qrm_solve!(spfct, ψ.res, y, transp='t')
  qrm_solve!(spfct, y, ψ.dsol, transp='n')

  ψ.sol .+= ψ.dsol

  ψ.sol .*= -1

  for i ∈ eachindex(ψ.sol)
    ψ.sol[i] = min(max(ψ.sol[i], - ψ.h.lambda * σ), ψ.h.lambda * σ)
  end

  mul!(y, ψ.A', ψ.sol)
  y .+= q

  return y
end