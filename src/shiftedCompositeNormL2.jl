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
  σ::R;
  tol = 1e-16,
  max_iter = 10000
) where {R <: Real, V0 <: Function,V1 <:Function,V2 <: AbstractMatrix{R}, V3 <: AbstractVector{R}, V4 <: AbstractVector{R}}
  
  if !ψ.is_shifted
    error("Shifted Norm L2 : Operator must be shifted for prox computation")
  end

  global α = 0.0
  g = ψ.A*q + ψ.b
  Δ = ψ.h.lambda*σ
  s = zero(g)
  m = length(g)

  try
    C = cholesky(ψ.A*ψ.A')
    s .=  C\(-g)
    if norm(s) <= Δ
      y .= q + ψ.A'*s
      return y
    end

    w = C.L\s
    α += ((norm(s)/norm(w))^2)*(norm(s)-Δ)/Δ

  catch ex 
    if isa(ex,LinearAlgebra.SingularException) || isa(ex,PosDefException)
      α_opt = sqrt(tol)
      while α <= 0 
        α_opt /= 10.0
        C = cholesky(ψ.A*ψ.A'+α*I(m))
        s .=  C\(-g)
        w = C.L\s
        α = α_opt + ((norm(s)/norm(w))^2)*(norm(s)-Δ)/Δ
      end
    else
      rethrow()
    end

  end
  
  α_hist = zeros(R,max_iter)
  k = 0

  while abs(norm(s)-Δ)>tol

    k = k + 1 
    C = cholesky(ψ.A*ψ.A'+α*I(m))
    s .=  C\(-g)
    w = C.L\s

    αn = ((norm(s)/norm(w))^2)*(norm(s)-Δ)/Δ
    α_hist[k] = αn

    
    if abs(αn) < tol
      if abs(norm(s)-Δ) < sqrt(tol)
        break
      else 
        error("Shifted Norm L2 : Newton method did not converge")
      end
    end

    if k >= max_iter
      for i = 1:max_iter-1 ## Check for oscillations in the Newton method
        if abs(α_hist[k]-α_hist[k-i]) < tol && abs(norm(s)-Δ) < sqrt(tol)
          break
        else 
          error("Shifted Norm L2 : Newton method did not converge")
        end
      end
    
    end
    α += αn

  end
  y .= q + ψ.A'*s

  return y

end