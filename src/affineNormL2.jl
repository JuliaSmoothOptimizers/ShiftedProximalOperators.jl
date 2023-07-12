# Composition of the L2 norm with an affine function
export AffineNormL2

@doc raw"""
    AffineNormL2(A,b,λ=1)

Returns the ``\ell_{2}`` norm operator composed with an affine function
```math
f(x) = λ ||Ax+b||_2
```
where ``\lambda > 0``, A is some mxn matrix and b is a vector in Rᵐ.
"""
struct AffineNormL2{R <: Real, V1 <: AbstractMatrix{R}, V2 <: AbstractVector{R}}
  lambda::R
  A::V1
  b::V2
  function AffineNormL2{R,V1,V2}(A::V1,b::V2,lambda::R) where {R <: Real, V1 <: AbstractMatrix{R}, V2 <: AbstractVector{R}}
    if lambda < 0
      error("Affine Norm L2 : parameter λ must be nonnegative")
    elseif size(A,1) != length(b)
      error("Affine Norm L2 : dimensions of parameter A and b must match")
    else
      new{R,V1,V2}(lambda,A,b)
    end
  end
end

AffineNormL2(A::V1,b::V2,lambda::R = 1.0) where {R <: Real, V1 <: AbstractMatrix{R},V2 <: AbstractVector{R}} = AffineNormL2{R,V1,V2}(A,b,lambda)

function (f::AffineNormL2)(x::AbstractVector{T}) where {T <: Real}
  if size(A,2) != length(x)
    error("Affine Norm L2 : dimensions of x does not match dimensions of A")
  end
  return f.lambda * norm(f.A*x + f.b)
end

function prox!(
  y::AbstractVector{T},
  f::AffineNormL2{R,V1,V2},
  x::AbstractVector{T},
  gamma::Real = 1,
) where {T <: Real, R <: Real, V1 <: AbstractMatrix{R}, V2 <: AbstractVector{R}}

try
  z = -f.A*f.A'\(f.A*x + f.b)
catch ex 
  if isa(ex,LinearAlgebra.SingularException)
    error("Affine Norm L2 : A is not full row rank !")
  else
    rethrow()
  end

end

if norm(z) <= f.lambda*gamma
  y .= x + f.A'*z
  return y
end

m = length(f.b)
g(α::T) = (z = (f.A*f.A' + α*I(m))\(f.A*x+f.b); norm(z)^2 - (f.lambda*gamma)^2)
Dg(α::T) = (z = (f.A*f.A' + α*I(m))^3\(f.A*x+f.b); -2*(f.A*x+f.b)'*z)
α_root = find_zero((g,Dg),0.0,Roots.Newton())

z = -(f.A*f.A' + α_root*I(m))\(f.A*x+f.b)
y .= x + f.A'*z

return y
end

fun_name(f::AffineNormL2) = "ℓ₂ norm of the affine function x ↦ Ax+b"
fun_dom(f::AffineNormL2) = "AbstractVector{Real}"
fun_expr(f::AffineNormL2{T,V1,V2}) where {T <: Real, V1 <: AbstractMatrix{T}, V2 <: AbstractVector{T}} = "x ↦ λ ‖Ax+b‖₂"
fun_params(f::AffineNormL2{T,V1,V2}) where {T <: Real, V1 <: AbstractMatrix{T}, V2 <: AbstractVector{T}} = "λ = $(f.lambda)\n A = $(f.A)\n b = $(f.b)"