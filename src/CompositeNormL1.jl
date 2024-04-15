# Composition of the L1 norm with a function
export CompositeNormL1

@doc raw"""
    CompositeNormL1(λ, c!, J!, A, b)

Returns the ``\ell_{1}`` norm operator composed with a function
```math
f(x) = λ \|c(x)\|_1
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
mutable struct CompositeNormL1{
    R <: Real,
    V0 <: Function,
    V1 <: Function,
    V2 <: AbstractMatrix{R},
    V3 <: AbstractVector{R},
  } <: AbstractCompositeNorm
    h::NormL1{R}
    c!::V0
    J!::V1
    A::V2
    b::V3

    function CompositeNormL1(
      λ::R,
      c!::Function,
      J!::Function,
      A::AbstractMatrix{R},
      b::AbstractVector{R},
    ) where {R <: Real}
      λ > 0 || error("CompositeNormL1: λ should be positive")
      length(b) == size(A,1) || error("Composite Norm L1 : Wrong input dimensions, c(x) should have same length as rows of J(x)")  
      new{R, typeof(c!), typeof(J!), typeof(A), typeof(b)}(NormL1(λ), c!, J!, A, b)
    end
  end

fun_name(f::CompositeNormL1) = "ℓ₁ norm of the function c"
fun_dom(f::CompositeNormL1) = "AbstractVector{Real}"
fun_expr(f::CompositeNormL1{T,V0,V1,V2,V3}) where {T <: Real,V0 <: Function, V1 <: Function, V2 <: AbstractMatrix{T}, V3 <: AbstractVector{T}} = "x ↦ λ ‖c(x)‖₁"
