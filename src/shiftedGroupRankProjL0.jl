export ShiftedGroupRankProjL0

mutable struct ShiftedGroupRankProjL0{
  R <: Real,
  RR <: AbstractVector{R},
  I,
  F,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::GroupRankProjL0{R, RR, I, F}
  xk::V0
  sj::V1
  sol::V2
  shifted_twice::Bool
  xsy::V2

  function ShiftedGroupRankProjL0(
    h::GroupRankProjL0{R, RR, I, F},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real, RR <: AbstractVector{R}, I, F}
    sol = similar(xk)
    xsy = similar(xk)
    new{R, RR, I, F, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice, xsy)
  end
end

shifted(
  h::GroupRankProjL0{R, RR, I, F},
  xk::AbstractVector{R},
) where {R <: Real, RR <: AbstractVector{R}, I, F} = ShiftedGroupRankProjL0(h, xk, zero(xk), false)

shifted(
  ψ::ShiftedGroupRankProjL0{R, RR, I, F, V0, V1, V2},
  sj::AbstractVector{R},
) where {
  R <: Real,
  RR <: AbstractVector{R},
  I,
  F,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} = ShiftedGroupRankProjL0(ψ.h, ψ.xk, sj, true)

fun_name(ψ::ShiftedGroupRankProjL0) = "shifted group ℓ₀ penalty with cardinality and sets"
fun_expr(ψ::ShiftedGroupRankProjL0) = "t ↦ Σᵢ λᵢ ‖xk + sj + t‖₀ + χ_{B ∩ C_{l,s}}"
fun_params(ψ::ShiftedGroupRankProjL0) =
  "l = $(ψ.h.l), s = $(ψ.h.s)\n" * " "^14 * "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14

# ψ(s) = h(xk + sj + s). The prox at q is prox_h(q + xk + sj), shifted back by
# xk + sj. A coordinate in no group is outside h, so that shift leaves it equal to q.
function prox!(
  y::AbstractVector{R},
  ψ::ShiftedGroupRankProjL0{R, RR, I, F, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {
  R <: Real,
  RR <: AbstractVector{R},
  I,
  F,
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
