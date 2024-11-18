# Composition of the L2 norm with a function
export CompositeNormL2

@doc raw"""
    CompositeNormL2(λ, c!, J!, A, b)

Returns a function ``c`` composed with the ``ℓ₂`` norm.
```math
f(x) = λ ‖c(x)‖₂
```
where ``λ > 0``. `c!` and `J!` should implement functions 
```math
c : ℜⁿ ↦ ℜᵐ,
```
```math
J : ℜⁿ ↦ ℜᵐˣⁿ,   
```
such that ``J`` is the Jacobian of ``c``. `A` and `b` should respectively be a matrix and a vector which can respectively store the values of ``J`` and ``c``.
`A` is expected to be sparse, `c!` and `J!` should have signatures
```
    c!(b <: AbstractVector{Real}, xk <: AbstractVector{Real})
    J!(A <: AbstractSparseMatrixCOO{Real,Integer}, xk <: AbstractVector{Real})
```
"""
mutable struct CompositeNormL2{
    R <: Real,
    V0 <: Function,
    V1 <: Function,
    V2 <: AbstractMatrix{R},
    V3 <: AbstractVector{R},
  } <: AbstractCompositeNorm
    h::NormL2{R}
    c!::V0
    J!::V1
    A::V2
    b::V3

    function CompositeNormL2(
      λ::R,
      c!::Function,
      J!::Function,
      A::AbstractMatrix{R},
      b::AbstractVector{R},
    ) where {R <: Real}
      λ > 0 || error("CompositeNormL2: λ should be positive")
      length(b) == size(A,1) || error("Composite Norm L2: Wrong input dimensions, c(x) should have same length as rows of J(x)")  
      new{R, typeof(c!), typeof(J!), typeof(A), typeof(b)}(NormL2(λ), c!, J!, A, b)
    end
  end

fun_name(f::CompositeNormL2) = "ℓ₂ norm of the function c"
fun_dom(f::CompositeNormL2) = "AbstractVector{Real}"
fun_expr(f::CompositeNormL2{T,V0,V1,V2,V3}) where {T <: Real,V0 <: Function, V1 <: Function, V2 <: AbstractMatrix{T}, V3 <: AbstractVector{T}} = "x ↦ λ ‖c(x)‖₂"