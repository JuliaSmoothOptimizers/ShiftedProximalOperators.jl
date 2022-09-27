# Nuclearnorm function

export Nuclearnorm

@doc raw"""
    Nuclearnorm(λ)

Returns the nuclear norm
```math
f(x) =  \lambda \|X\|_*
```
for a nonnegative parameter ``\lambda`` and a vector ``x``, where
``x = \text{vec}(X)``.
"""
mutable struct Nuclearnorm{R <: Real, S <: AbstractArray, T, Tr, M <: AbstractArray{T}}
  lambda::R
  A::S
  F::PSVD{T, Tr, M}
  function Nuclearnorm(
    lambda::R,
    A::S,
    F::PSVD{T, Tr, M},
  ) where {R <: Real, S <: AbstractArray, T, Tr, M <: AbstractArray{T}}
    if lambda < 0
      error("λ must be nonnegative")
    end
    new{typeof(lambda), typeof(A), T, Tr, M}(lambda, A, F)
  end
end

Nuclearnorm(lambda::R, A::S, F::PSVD{T, Tr, M}) where {R, S, T, Tr, M <: AbstractArray{T}} =
  Nuclearnorm{R, S, eltype(A), real(eltype(A)), M}(lambda, A, F)

Nuclearnorm(lambda::R, A::S) where {R, S} = begin
  F = psvd_workspace_dd(A, full = false)
  Nuclearnorm(lambda, A, F)
end

function (f::Nuclearnorm)(x::AbstractVector{R}) where {R <: Real}
  f.A .= reshape_array(x, size(f.A))
  psvd_dd!(f.F, f.A, full = false)
  return f.lambda * sum(f.F.S)
end

fun_name(f::Nuclearnorm) = "Nuclearnorm"
fun_dom(f::Nuclearnorm) = "AbstractArray{Real}"
fun_expr(f::Nuclearnorm{T}) where {T <: Real} = "x ↦ Nuclearnorm(matrix(x))"
fun_params(f::Nuclearnorm{T}) where {T <: Real} = "λ = $(f.lambda)"

function prox!(
  y::AbstractVector{R},
  f::Nuclearnorm{R, S, T, Tr, M},
  x::AbstractVector{R},
  gamma::R,
) where {R <: Real, S <: AbstractArray, T, Tr, M <: AbstractArray{T}}
  f.A .= reshape_array(x, size(f.A))
  psvd_dd!(f.F, f.A, full = false)
  c = sqrt(2 * f.lambda * gamma)
  f.F.S .= max.(0, f.F.S .- f.lambda * gamma)
  for i ∈ eachindex(f.F.S)
    for j = 1:size(f.A, 1)
      f.F.U[j, i] = f.F.U[j, i] * f.F.S[i]
    end
  end
  mul!(f.A, f.F.U, f.F.Vt)
  y .= reshape_array(f.A, (size(y, 1), 1))
  return y
end
