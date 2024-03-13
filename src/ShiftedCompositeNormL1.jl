export ShiftedCompositeNormL1

mutable struct ShiftedCompositeNormL1{
  R <: Real,
  V0 <: Function,
  V1 <: Function,
  V2 <: AbstractMatrix{R},
  V3 <: AbstractVector{R},
  V4 <: AbstractVector{R}
} <: ShiftedCompositeProximableFunction
  h::NormL1{R}
  c!::V0
  J!::V1
  A::V2
  b::V3
  sol::V4
  function ShiftedCompositeNormL1(
    h::NormL1{R},
    c!::Function,
    J!::Function,
    A::AbstractMatrix{R},
    b::AbstractVector{R},
  ) where {R <: Real}
    sol = similar(b,size(A,2))
    if length(b) != size(A,1)
      error("ShiftedCompositeNormL1 : Wrong input dimensions, constraints should have same length as rows of the Jacobian")
    end
    new{R,typeof(c!),typeof(J!),typeof(A),typeof(b), typeof(sol)}(h,c!,J!,A,b, sol)
  end
end


shifted(h::NormL1{R}, c!::Function,J!::Function,A::AbstractMatrix{R},b::AbstractVector{R}, xk :: AbstractVector{R}) where {R <: Real} = begin
  c!(b,xk)
  J!(A,xk)
  ShiftedCompositeNormL1(h,c!,J!,A,b)
end

shifted(
  ψ::ShiftedCompositeNormL1{R, V0, V1, V2, V3, V4},
  xk::AbstractVector{R},
) where {R <: Real, V0 <: Function, V1 <: Function, V2 <: AbstractMatrix{R},V3<: AbstractVector{R},V4 <: AbstractVector{R}} = begin
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
  ψ::ShiftedCompositeNormL1{R, V0, V1, V2, V3, V4},
  q::AbstractVector{R},
  σ::R
) where {R <: Real, V0 <: Function,V1 <:Function,V2 <: AbstractMatrix{R}, V3 <: AbstractVector{R}, V4 <: AbstractVector{R}}
  
  g = ψ.A*q + ψ.b
  H = ψ.A*ψ.A'
  
  C = ldlt(H)
  s =  C\(-g)

  for i ∈ eachindex(s)
    s[i] = min(max(s[i], - ψ.h.lambda * σ), ψ.h.lambda * σ)
  end
  y .= q + ψ.A'*s

  return y 
end