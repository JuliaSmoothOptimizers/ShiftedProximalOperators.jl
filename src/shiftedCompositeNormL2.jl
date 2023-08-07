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
  function ShiftedCompositeNormL2(
    h::NormL2{R},
    c!::Function,
    J!::Function,
    A::AbstractMatrix{R},
    b::AbstractVector{R},
  ) where {R <: Real}
    sol = similar(b,size(A,2))
    if length(b) != size(A,1)
      error("Shifted Norm L2 : Wrong input dimensions, constraints should have same length as rows of the jacobian")
    end
    new{R,typeof(c!),typeof(J!),typeof(A),typeof(b), typeof(sol)}(h,c!,J!,A,b, sol)
  end
end


shifted(h::NormL2{R}, c!::Function,J!::Function,A::AbstractMatrix{R},b::AbstractVector{R}, xk :: AbstractVector{R}) where {R <: Real} = begin
  c!(b,xk)
  J!(A,xk)
  ShiftedCompositeNormL2(h,c!,J!,A,b)
end

shifted(
  ψ::ShiftedCompositeNormL2{R, V0, V1, V2, V3, V4},
  xk::AbstractVector{R},
) where {R <: Real, V0 <: Function, V1 <: Function, V2 <: AbstractMatrix{R},V3<: AbstractVector{R},V4 <: AbstractVector{R}} = begin
  b = similar(ψ.b)
  ψ.c!(b,xk)
  A = similar(ψ.A)
  ψ.J!(A,xk)
  ShiftedCompositeNormL2(ψ.h, ψ.c!,ψ.J!,A,b)
end
 
shifted(
  ψ::CompositeNormL2{R,V0,V1,V2,V3},
  xk::AbstractVector{R}
) where {R <: Real, V0 <: Function, V1 <: Function, V2 <: AbstractMatrix{R}, V3 <: AbstractVector{R}} = begin
  b = similar(ψ.b)
  ψ.c!(b,xk)
  A = similar(ψ.A)
  ψ.J!(A,xk)
  ShiftedCompositeNormL2(ψ.h,ψ.c!,ψ.J!,A,b)
end

fun_name(ψ::ShiftedCompositeNormL2) = "shifted L2 norm"
fun_expr(ψ::ShiftedCompositeNormL2) = "t ↦ ‖c(xk) + J(xk)t‖₂"
fun_params(ψ::ShiftedCompositeNormL2) = "c(xk) = $(ψ.b)\n" * " "^14 * "J(xk) = $(ψ.A)\n"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedCompositeNormL2{R, V0, V1, V2, V3, V4},
  q::AbstractVector{R},
  σ::R;
  tol = 1e-16,
  max_iter = 10000,
  debug = false
) where {R <: Real, V0 <: Function,V1 <:Function,V2 <: AbstractMatrix{R}, V3 <: AbstractVector{R}, V4 <: AbstractVector{R}}
  

  α = 0.0
  g = ψ.A*q + ψ.b
  H = ψ.A*ψ.A'

  Δ = ψ.h.lambda*σ
  s = zero(g)
  m = length(g)
  k = 0

  try
    C = cholesky(H)
    s .=  C\(-g)
    if norm(s) <= Δ
      y .= q + ψ.A'*s
      return y
    end

    w = C.L\s
    α += ((norm(s)/norm(w))^2)*(norm(s)-Δ)/Δ

  catch ex 
    if isa(ex,LinearAlgebra.SingularException) || isa(ex,PosDefException)
      α_opt = 10.0*sqrt(tol)
      while α <= 0 
        α_opt /= 10.0
        C = cholesky(H+α_opt*I(m))
        s .=  C\(-g)
        w = C.L\s
        α = α_opt + ((norm(s)/norm(w))^2)*(norm(s)-Δ)/Δ
      end
    else  
      rethrow()
    end

  end
  

  # Cf Algorithm 7.3.1 in Conn-Gould-Toint
  while abs(norm(s)-Δ)>tol && k < max_iter

    k = k + 1 
    if debug 
      println(α)
      println(norm(s)-Δ )
    end
    C = cholesky(H+α*I(m))
    s .=  C\(-g)
    w = C.L\s

    αn = ((norm(s)/norm(w))^2)*(norm(s)-Δ)/Δ
    α = max(0,α+αn)

    if abs(αn) < tol &&  abs(norm(s)-Δ)<sqrt(tol)
      break
    end

  end
  y .= q + ψ.A'*s

  if k > max_iter && abs(norm(s)-Δ)>sqrt(tol)
    error("ShiftedCompositeNormL2 : Newton Method did not converge.")
  end 

  return y
end