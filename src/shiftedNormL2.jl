export ShiftedNormL2

mutable struct ShiftedNormL2{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: Function,
  V3 <: Function,
  V4 <: AbstractMatrix{R},
  V5 <: AbstractVector{R},
  V6 <: AbstractVector{R}
} <: ShiftedCompositeProximableFunction
  h::NormL2{R}
  xk::V0
  sj::V1
  c!::V2
  J!::V3
  A::V4
  b::V5
  sol::V6
  shifted_twice::Bool
  function ShiftedNormL2(
    h::NormL2{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    c!::Function,
    J!::Function,
    A::AbstractMatrix{R},
    b::AbstractVector{R},
    shifted_twice::Bool
  ) where {R <: Real}
    sol = similar(xk)
    J!(A,xk+sj)
    c!(b,xk+sj)
    new{R, typeof(xk), typeof(sj),typeof(c!),typeof(J!),typeof(A),typeof(b), typeof(sol)}(h, xk, sj, c!,J!,A,b, sol, shifted_twice)
  end
end


shifted(h::NormL2{R}, xk::AbstractVector{R}, c!::Function,J!::Function,A::AbstractMatrix{R},b::AbstractVector{R}) where {R <: Real} =
  ShiftedNormL2(h, xk, zero(xk), c!, J!,A,b, false)
shifted(
  ψ::ShiftedNormL2{R, V0, V1, V2, V3, V4, V5,V6},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: Function, V3 <: Function, V4 <: AbstractMatrix{R},V5<: AbstractVector{R},V6<: AbstractVector{R}} =
  ShiftedNormL2(ψ.h, ψ.xk,sj, ψ.c! , ψ.J!,ψ.A,ψ.b, true)

fun_name(ψ::ShiftedNormL2) = "shifted L2 norm"
fun_expr(ψ::ShiftedNormL2) = "t ↦ ‖c(xk + sk) + J(xk+sj)t‖₂"
fun_params(ψ::ShiftedNormL2) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL2{R, V0, V1, V2, V3, V4,V5,V6},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: Function,V3 <:Function,V4 <: AbstractMatrix{R}, V5 <: AbstractVector{R}, V6 <: AbstractVector{R}}
  
  b = ψ.b
  if ψ.h.lambda^(-1)*ψ.h(inv(ψ.A*ψ.A')*(ψ.A*q + b)) <= ψ.h.lambda*σ 
    y .= q - ψ.A'*inv(ψ.A*ψ.A')*(ψ.A*q+b)
    return y
  end
  m = length(b)
  f(x) = ψ.h.lambda^(-2)*ψ.h(inv(ψ.A*ψ.A' + x*I(m))*(ψ.A*q+b))^2 - (ψ.h.lambda*σ)^2
  Df(x) = -2*(ψ.A*q+b)'*inv(ψ.A*ψ.A'+x*I(m))^3*(ψ.A*q+b)
  α = find_zero((f,Df),0.0,Roots.Newton())
  
  y .= q - ψ.A'*inv(ψ.A*ψ.A'+α*I(m))*(ψ.A*q+b)

  return y


  
end