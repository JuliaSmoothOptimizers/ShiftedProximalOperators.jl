export ShiftedNormL2

mutable struct ShiftedNormL2{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractMatrix{R},
  V3 <: AbstractVector{R}
} <: ShiftedLinearProximableFunction
  h::NormL2{R}
  xk::V0
  sj::V1
  A::V2
  sol::V3
  shifted_twice::Bool
  function ShiftedNormL2(
    h::NormL2{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    A::AbstractMatrix{R},
    shifted_twice::Bool
  ) where {R <: Real}
    sol = similar(xk)
    new{R, typeof(xk), typeof(sj),typeof(A), typeof(sol)}(h, xk, sj, A, sol, shifted_twice)
  end
end


shifted(h::NormL2{R}, xk::AbstractVector{R}, A::AbstractMatrix{R}) where {R <: Real} =
  ShiftedNormL2(h, xk, zero(xk), A, false)
shifted(
  ψ::ShiftedNormL2{R, V0, V1, V2, V3},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractMatrix{R}, V3 <: AbstractArray{R}} =
  ShiftedNormL2(ψ.h, ψ.xk, sj, ψ.A, true)

fun_name(ψ::ShiftedNormL2) = "shifted L2 norm"
fun_expr(ψ::ShiftedNormL2) = "t ↦ ‖xk + sk + At‖₂"
fun_params(ψ::ShiftedNormL2) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n"* " "^14 *"A = $(ψ.A)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL2{R, V0, V1, V2, V3},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractMatrix{R}, V3 <: AbstractVector{R}}
  

  if ψ.h.lambda^(-1)*ψ.h(inv(ψ.A*ψ.A')*(ψ.A*q+ψ.xk+ψ.sj)) <= ψ.h.lambda*σ 
    y .= q - ψ.A'*inv(ψ.A*ψ.A')*(ψ.A*q+ψ.xk+ψ.sj)
    return y
  end
  m = length(ψ.xk)
  f(x) = ψ.h.lambda^(-2)*ψ.h(inv(ψ.A*ψ.A' + x*I(m))*(ψ.A*q+ψ.xk+ψ.sj))^2 - (ψ.h.lambda*σ)^2
  Df(x) = -2*(ψ.A*q+ψ.xk + ψ.sj)'*inv(ψ.A*ψ.A'+x*I(m))^3*(ψ.A*q+ψ.xk + ψ.sj)
  α = find_zero((f,Df),0.0,Roots.Newton())
  
  y .= q - ψ.A'*inv(ψ.A*ψ.A'+α*I(m))*(ψ.A*q+ψ.xk+ψ.sj)

  return y


  
end