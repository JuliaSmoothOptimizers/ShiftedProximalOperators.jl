export ShiftedNormL0

mutable struct ShiftedNormL0{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::NormL0{R}
  xk::V0  # base shift (nonzero when shifting an already shifted function)
  sj::V1  # current shift
  sol::V2   # internal storage
  shifted_twice::Bool
  function ShiftedNormL0(
    h::NormL0{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(xk)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice)
  end
end

fun_name(ψ::ShiftedNormL0) = "shifted L0 pseudo-norm"
fun_expr(ψ::ShiftedNormL0) = "t ↦ ‖xk + sj + t‖₀"

shifted(h::NormL0{R}, xk::AbstractVector{R}) where {R <: Real} =
  ShiftedNormL0(h, xk, zero(xk), false)
shifted(
  ψ::ShiftedNormL0{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormL0(ψ.h, ψ.xk, sj, true)

function prox(
  ψ::ShiftedNormL0{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  c = sqrt(2 * ψ.λ * σ)
  for i ∈ eachindex(q)
    xps = ψ.xk[i] + ψ.sj[i]
    if abs(xps + q[i]) ≤ c
      ψ.sol[i] = -xps
    else
      ψ.sol[i] = q[i]
    end
  end
  return ψ.sol
end
