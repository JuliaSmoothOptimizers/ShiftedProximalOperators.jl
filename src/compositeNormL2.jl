# Composition of the L2 norm with a function
export CompositeNormL2

@doc raw"""
    CompositeNormL2(λ, c!, J!, A, b)

Returns function `c` composed with the `ℓ₂` norm:
```math
f(x) = λ ‖c(x)‖₂
```
where `λ > 0`. `c!` and `J!` should implement functions 
```math
c : ℝⁿ ↦ ℝᵐ,
```
```math
J : ℝⁿ ↦ ℝᵐˣⁿ,   
```
such that `J` is the Jacobian of `c`. It is expected that `m ≤ n`.
`A` and `b` should respectively be a matrix and a vector which can respectively store the values of `J` and `c`.
`A` is expected to be sparse, `c!` and `J!` should have signatures
```
    c!(b <: AbstractVector{Real}, xk <: AbstractVector{Real})
    J!(A <: AbstractSparseMatrixCOO{Real, Integer}, xk <: AbstractVector{Real})
```
"""
mutable struct CompositeNormL2{
    T <: Real,
    F0 <: Function,
    F1 <: Function,
    M <: AbstractMatrix{T},
    V <: AbstractVector{T},
  } <: AbstractCompositeNorm
    h::NormL2{T}
    c!::F0
    J!::F1
    A::M
    b::V
    g::V

    function CompositeNormL2(
      λ::T,
      c!::Function,
      J!::Function,
      A::AbstractMatrix{T},
      b::AbstractVector{T},
    ) where {T <: Real}
      λ > 0 || error("CompositeNormL2: λ should be positive")
      length(b) == size(A, 1) || error("Composite Norm L2: Wrong input dimensions, the length of c(x) should be the same as the number of rows of J(x)")  
      new{T, typeof(c!), typeof(J!), typeof(A), typeof(b)}(NormL2(λ), c!, J!, A, b, similar(b))
    end
  end

fun_name(f::CompositeNormL2) = "ℓ₂ norm of the function c"
fun_dom(f::CompositeNormL2) = "AbstractVector{Real}"
fun_expr(f::CompositeNormL2{T, F0, F1, M, V}) where {T <: Real, F0 <: Function, F1 <: Function, M <: AbstractMatrix{T}, V <: AbstractVector{T}} = "x ↦ λ ‖c(x)‖₂"