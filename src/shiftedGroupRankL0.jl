export ShiftedGroupRankL0

mutable struct ShiftedGroupRankL0{
  R <: Real,
  RR <: AbstractVector{R},
  I,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::GroupRankL0{R, RR, I}
  xk::V0
  sj::V1
  sol::V2
  shifted_twice::Bool
  xsy::V2

  function ShiftedGroupRankL0(
    h::GroupRankL0{R, RR, I},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real, RR <: AbstractVector{R}, I}
    sol = similar(xk)
    xsy = similar(xk)
    new{R, RR, I, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice, xsy)
  end
end

shifted(h::GroupRankL0{R, RR, I}, xk::AbstractVector{R}) where {R <: Real, RR <: AbstractVector{R}, I} =
  ShiftedGroupRankL0(h, xk, zero(xk), false)

shifted(
  ψ::ShiftedGroupRankL0{R, RR, I, V0, V1, V2},
  sj::AbstractVector{R},
) where {
  R <: Real,
  RR <: AbstractVector{R},
  I,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} = ShiftedGroupRankL0(ψ.h, ψ.xk, sj, true)

fun_name(ψ::ShiftedGroupRankL0) = "shifted group ℓ₀ penalty with cardinality"
fun_expr(ψ::ShiftedGroupRankL0) = "t ↦ Σᵢ λᵢ ‖xk + sj + t‖₀ + χ_{C_{l,s}}"
fun_params(ψ::ShiftedGroupRankL0) =
  "l = $(ψ.h.l), s = $(ψ.h.s)\n" * " "^14 * "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14

# ψ(s) = h(xk + sj + s). The prox at q is prox_h(q + xk + sj), shifted back by
# xk + sj. A coordinate in no group is outside h, so that shift leaves it equal to q.
function prox!(
  y::AbstractVector{R},
  ψ::ShiftedGroupRankL0{R, RR, I, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {
  R <: Real,
  RR <: AbstractVector{R},
  I,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
}
  # A coordinate in no group is outside h, so this shift leaves it equal to q.
  @. ψ.sol = q + ψ.xk + ψ.sj
  val = prox!(y, ψ.h, ψ.sol, σ)
  @. y -= ψ.xk + ψ.sj
  return val
end
