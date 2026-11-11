export ShiftedGroupNormL0

mutable struct ShiftedGroupNormL0{
  R <: Real,
  RR <: AbstractVector{R},
  I,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::GroupNormL0{R, RR, I}
  xk::V0
  sj::V1
  sol::V2
  shifted_twice::Bool
  xsy::V2

  function ShiftedGroupNormL0(
    h::GroupNormL0{R, RR, I},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real, RR <: AbstractVector{R}, I}
    sol = similar(xk)
    xsy = similar(xk)
    new{R, RR, I, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice, xsy)
  end
end

shifted(h::GroupNormL0{R, RR, I}, xk::AbstractVector{R}) where {R <: Real, RR <: AbstractVector{R}, I} =
  ShiftedGroupNormL0(h, xk, zero(xk), false)

shifted(
  ψ::ShiftedGroupNormL0{R, RR, I, V0, V1, V2},
  sj::AbstractVector{R},
) where {
  R <: Real,
  RR <: AbstractVector{R},
  I,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} = ShiftedGroupNormL0(ψ.h, ψ.xk, sj, true)

fun_name(ψ::ShiftedGroupNormL0) = "shifted group ℓ₀ penalty"
fun_expr(ψ::ShiftedGroupNormL0) = "t ↦ Σᵢ λᵢ ‖xk + sj + t‖₀ on each group"
fun_params(ψ::ShiftedGroupNormL0) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14

# ψ(s) = h(xk + sj + s). The prox at q is prox_h(q + xk + sj), shifted back by
# xk + sj. A coordinate in no group is outside h, so that shift leaves it equal to q.
function prox!(
  y::AbstractVector{R},
  ψ::ShiftedGroupNormL0{R, RR, I, V0, V1, V2},
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
