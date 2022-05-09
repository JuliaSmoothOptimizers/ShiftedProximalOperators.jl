export ShiftedNormL0

mutable struct ShiftedNormL0{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3 <: AbstractVector{R}
} <: ShiftedProximableFunction
  h::NormL0{R}
  xk::V0  # base shift (nonzero when shifting an already shifted function)
  sj::V1  # current shift
  sol::V2   # internal storage
  selected::V3 # used for selected index
  shifted_twice::Bool
  function ShiftedNormL0(
    h::NormL0{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    selected::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(xk)
    new{R, typeof(xk), typeof(sj),typeof(selected) ,typeof(sol)}(h, xk, sj,selected ,sol, shifted_twice)
  end
end

fun_name(ψ::ShiftedNormL0) = "shifted L0 pseudo-norm"
fun_expr(ψ::ShiftedNormL0) = "t ↦ ‖xk + sj + t‖₀"

shifted(h::NormL0{R}, xk::AbstractVector{R}) where {R <: Real} =
  ShiftedNormL0(h, xk, zero(xk),[0:size(xk)[1];], false)
shifted(
  ψ::ShiftedNormL0{R, V0, V1, V2,V3},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R},V3 <: AbstractVector{R}} =
  ShiftedNormL0(ψ.h, ψ.xk, sj, selected,true)

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL0{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  c = sqrt(2 * ψ.λ * σ)
  for i ∈ eachindex(q)
    if i ∈ ψ.selected
      xps = ψ.xk[i] + ψ.sj[i]
      if abs(xps + q[i]) ≤ c
        y[i] = -xps
      else
        y[i] = q[i]
      end
    else
      y[i] = q[i]
    end
  end
  return y
end
