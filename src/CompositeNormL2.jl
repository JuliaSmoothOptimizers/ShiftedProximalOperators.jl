# Composition of the L2 norm with a function
export CompositeNormL2

@doc raw"""
    CompositeNormL2(h,c!,J!,A,b)

Returns the ``\ell_{2}`` norm operator composed with a function
```math
f(x) = λ ||c(x)||_2
```
where ``\lambda > 0``.
"""
mutable struct CompositeNormL2{
    R <: Real,
    V0 <: Function,
    V1 <: Function,
    V2 <: AbstractMatrix{R},
    V3 <: AbstractVector{R},
  } <: CompositeProximableFunction
    h::NormL2{R}
    c!::V0
    J!::V1
    A::V2
    b::V3
    function CompositeNormL2(
      h::NormL2{R},
      c!::Function,
      J!::Function,
      A::AbstractMatrix{R},
      b::AbstractVector{R},
    ) where {R <: Real}
      length(b) == size(A,1) || error("Composite Norm L2 : Wrong input dimensions, constraints should have same length as rows of the jacobian")  
      new{R,typeof(c!),typeof(J!),typeof(A),typeof(b)}(h,c!,J!,A,b)
    end
  end

fun_name(f::CompositeNormL2) = "ℓ₂ norm of the function x ↦ c(x)"
fun_dom(f::CompositeNormL2) = "AbstractVector{Real}"
fun_expr(f::CompositeNormL2{T,V0,V1,V2,V3}) where {T <: Real,V0 <: Function, V1 <: Function, V2 <: AbstractMatrix{T}, V3 <: AbstractVector{T}} = "x ↦ λ ‖c(x)‖₂"
