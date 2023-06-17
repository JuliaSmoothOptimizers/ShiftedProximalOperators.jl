export ShiftedNormL2Transform

mutable struct ShiftedNormL2Transform{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractMatrix{R},
  V3 <: AbstractMatrix{R},
  V4 <: AbstractVector{R}
} <: ShiftedProximableFunction
  h::NormL2{R}
  xk::V0
  sj::V1
  A::V2
  Q::V3
  sol::V4
  shifted_twice::Bool
  function ShiftedNormL2Transform(
    h::NormL2{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    A::AbstractMatrix{R},
    Q::AbstractMatrix{R},
    shifted_twice::Bool
  ) where {R <: Real}
    sol = similar(xk)
    new{R, typeof(xk), typeof(sj),typeof(A),typeof(Q), typeof(sol)}(h, xk, sj, A,Q, sol, shifted_twice)
  end
end


shifted(h::NormL2{R}, xk::AbstractVector{R}, A::AbstractMatrix{R},Q::AbstractMatrix{R}) where {R <: Real} =
  ShiftedNormL2Transform(h, xk, zero(xk), A, Q, false)
shifted(
  ψ::ShiftedNormL2Transform{R, V0, V1, V2, V3, V4},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractMatrix{R}, V3 <: AbstractMatrix{R}, V4 <: AbstractVector{R}} =
  ShiftedNormL2Transform(ψ.h, ψ.xk, sj, ψ.A, ψ.Q, true)

fun_name(ψ::ShiftedNormL2Transform) = "shifted L2 norm transform"
fun_expr(ψ::ShiftedNormL2Transform) = "t ↦ tQt/2 - t.q + ‖xk + sk + At‖₂"
fun_params(ψ::ShiftedNormL2Transform) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n"* " "^14 *"A = $(ψ.A)\n" * " "^14*"$(ψ.Q)\n"
 
function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL2Transform{R, V0, V1, V2, V3,V4},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractMatrix{R}, V3 <: AbstractMatrix{R},V4 <:AbstractVector{R}}
  
  Qinv = inv(ψ.Q)
  if ψ.h.lambda^(-1)*ψ.h(inv(ψ.A*Qinv*ψ.A')*(ψ.A*Qinv*q+ψ.xk+ψ.sj)) <= ψ.h.lambda*σ 
    y .= Qinv*(q - ψ.A'*inv(ψ.A*Qinv*ψ.A')*(ψ.A*Qinv*q+ψ.xk+ψ.sj))
    return y
  end
  m = length(ψ.xk)
  f(x) = ψ.h.lambda^(-2)*ψ.h(inv(ψ.A*Qinv*ψ.A' + x*I(m))*(ψ.A*Qinv*q+ψ.xk+ψ.sj))^2 - (ψ.h.lambda*σ)^2
  Df(x) = -2*(ψ.A*Qinv*q+ψ.xk + ψ.sj)'*inv(ψ.A*Qinv*ψ.A'+x*I(m))^3*(ψ.A*Qinv*q+ψ.xk + ψ.sj)
  α = find_zero((f,Df),0.0,Roots.Newton())
  
  y .= Qinv*(q - ψ.A'*inv(ψ.A*Qinv*ψ.A'+α*I(m))*(ψ.A*Qinv*q+ψ.xk+ψ.sj))

  return y


  
end