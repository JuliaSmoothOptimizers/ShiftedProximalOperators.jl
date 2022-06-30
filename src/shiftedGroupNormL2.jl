export ShiftedGroupNormL2

mutable struct ShiftedGroupNormL2{
  R <: Real,
  RR <: AbstractVector{R},
  I,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::GroupNormL2{R, RR, I}
  xk::V0
  sj::V1
  sol::V2
  shifted_twice::Bool

  function ShiftedGroupNormL2(
    h::GroupNormL2{R, RR, I},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real, RR <: AbstractVector{R}, I}
    sol = similar(sj)
    new{R, RR, I, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice)
  end
end

shifted(
  h::GroupNormL2{R, RR, I},
  xk::AbstractVector{R},
) where {R <: Real, RR <: AbstractVector{R}, I} = ShiftedGroupNormL2(h, xk, zero(xk), false)
shifted(h::NormL2{R}, xk::AbstractVector{R}) where {R <: Real} =
  ShiftedGroupNormL2(GroupNormL2([h.lambda]), xk, zero(xk), false)
shifted(
  ψ::ShiftedGroupNormL2{R, RR, I, V0, V1, V2},
  sj::AbstractVector{R},
) where {
  R <: Real,
  RR <: AbstractVector{R},
  I,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} = ShiftedGroupNormL2(ψ.h, ψ.xk, sj, true)

fun_name(ψ::ShiftedGroupNormL2) = "shifted ∑ᵢ‖⋅‖₂ norm"
fun_expr(ψ::ShiftedGroupNormL2) = "t ↦ ∑ᵢ ‖xk + sj + t‖₂"
fun_params(ψ::ShiftedGroupNormL2) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedGroupNormL2{R, RR, I, V0, V1, V2},
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
  ψ.sol .= q + ψ.xk + ψ.sj
  for (idx, λ) ∈ zip(ψ.h.idx, ψ.h.lambda)
    snorm = norm(ψ.sol[idx])
    if snorm == 0
      y[idx] .= 0
    else
      y[idx] .= max(1 - σ * λ / snorm, 0) .* ψ.sol[idx]
    end
  end
  y .-= (ψ.xk + ψ.sj)
  return y
end
