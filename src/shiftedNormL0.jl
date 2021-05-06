export ShiftedNormL0

mutable struct ShiftedNormL0{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::NormL0{R}
  x0::V0  # base shift (nonzero when shifting an already shifted function)
  x::V1   # current shift
  s::V2   # internal storage
  function ShiftedNormL0(
    h::NormL0{R},
    x0::AbstractVector{R},
    x::AbstractVector{R},
  ) where {R <: Real}
    s = similar(x)
    new{R, typeof(x0), typeof(x), typeof(s)}(h, x0, x, s)
  end
end

fun_name(ψ::ShiftedNormL0) = "shifted L0 pseudo-norm"
fun_expr(ψ::ShiftedNormL0) = "s ↦ ‖x + s‖₀"

shifted(h::NormL0{R}, x::AbstractVector{R}) where {R <: Real} = ShiftedNormL0(h, zero(x), x)
shifted(
  ψ::ShiftedNormL0{R, V0, V1, V2},
  x::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormL0(ψ.h, ψ.x, x)

function prox(
  ψ::ShiftedNormL0{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  c = sqrt(2 * ψ.λ * σ)
  for i ∈ eachindex(q)
    xpq = ψ.x0[i] + ψ.x[i] + q[i]
    if abs(xpq) ≤ c
      ψ.s[i] = 0
    else
      ψ.s[i] = xpq
    end
  end
  ψ.s .-= ψ.x0 .+ ψ.x
  return ψ.s
end
