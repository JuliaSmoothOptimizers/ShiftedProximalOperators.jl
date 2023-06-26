export ShiftedCompositeNormL2

mutable struct ShiftedCompositeNormL2{
  R <: Real,
  V0 <: Function,
  V1 <: Function,
  V2 <: AbstractMatrix{R},
  V3 <: AbstractVector{R},
  V4 <: AbstractVector{R}
} <: ShiftedCompositeProximableFunction
  h::NormL2{R}
  c!::V0
  J!::V1
  A::V2
  b::V3
  sol::V4
  is_shifted::Bool
  function ShiftedCompositeNormL2(
    h::NormL2{R},
    c!::Function,
    J!::Function,
    A::AbstractMatrix{R},
    b::AbstractVector{R},
    is_shifted::Bool
  ) where {R <: Real}
    sol = similar(b,size(A,2))
    if length(b) != size(A,1)
      error("Shifted Norm L2 : Wrong input dimensions, constraints should have same length as rows of the jacobian")
    end
    new{R,typeof(c!),typeof(J!),typeof(A),typeof(b), typeof(sol)}(h,c!,J!,A,b, sol,is_shifted)
  end
end

shifted(h::NormL2{R}, c!::Function,J!::Function,A::AbstractMatrix{R},b::AbstractVector{R}) where {R <: Real} = 
  ShiftedCompositeNormL2(h,c!,J!,A,b,false)
shifted(h::NormL2{R}, c!::Function,J!::Function,A::AbstractMatrix{R},b::AbstractVector{R}, xk :: AbstractVector{R}) where {R <: Real} =
  (c!(xk,b);J!(xk,A);ShiftedCompositeNormL2(h,c!,J!,A,b,true))
shifted(
  ψ::ShiftedCompositeNormL2{R, V0, V1, V2, V3, V4},
  xk::AbstractVector{R},
) where {R <: Real, V0 <: Function, V1 <: Function, V2 <: AbstractMatrix{R},V3<: AbstractVector{R},V4<: AbstractVector{R}} =
  (b = similar(ψ.b);ψ.c!(xk,b);A = similar(ψ.A);ψ.J!(xk,A);ShiftedCompositeNormL2(ψ.h, ψ.c!,ψ.J!,A,b,true)) 

fun_name(ψ::ShiftedCompositeNormL2) = "shifted L2 norm"
fun_expr(ψ::ShiftedCompositeNormL2) = "t ↦ ‖c(xk) + J(xk)t‖₂"
fun_params(ψ::ShiftedCompositeNormL2) = "c(xk) = $(ψ.b)\n" * " "^14 * "J(xk) = $(ψ.A)\n"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedCompositeNormL2{R, V0, V1, V2, V3, V4},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: Function,V1 <:Function,V2 <: AbstractMatrix{R}, V3 <: AbstractVector{R}, V4 <: AbstractVector{R}}
  
  if !ψ.is_shifted
    error("Shifted Norm L2 : Operator must be shifted for prox computation")
  end
  b = ψ.b

  try
    z = -ψ.A*ψ.A'\(ψ.A*q + b)
  catch ex 
    if isa(ex,LinearAlgebra.SingularException)
      error("Shifted Norm L2 : Jacobian is not full row rank")
    else
      rethrow()
    end

  end
  
  if ψ.h.lambda^(-1)*ψ.h(z) <= ψ.h.lambda*σ 
    y .= q + ψ.A'*z
    return y
  end

  m = length(b)
  f(x::R) = (z = (ψ.A*ψ.A' + x*I(m))\(ψ.A*q+b); ψ.h.lambda^(-2)*ψ.h(z)^2 - (ψ.h.lambda*σ)^2)
  Df(x::R) = (z = (ψ.A*ψ.A' + x*I(m))^3\(ψ.A*q+b); -2*(ψ.A*q+b)'*z)
  α = find_zero((f,Df),0.0,Roots.Newton())
  
  z = -(ψ.A*ψ.A' + α*I(m))\(ψ.A*q+b)
  y .= q + ψ.A'*z

  return y

end