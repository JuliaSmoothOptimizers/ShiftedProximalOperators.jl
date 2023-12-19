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
    sol = similar(sj)
    xsy = similar(sj)
    new{R, RR, I, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice, xsy)
  end
end

shifted(
  h::GroupNormL0{R, RR, I},
  xk::AbstractVector{R},
) where {R <: Real, RR <: AbstractVector{R}, I} = ShiftedGroupNormL0(h, xk, zero(xk), false)
shifted(h::NormL2{R}, xk::AbstractVector{R}) where {R <: Real} =
  ShiftedGroupNormL0(GroupNormL0([h.lambda]), xk, zero(xk), false)
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

fun_name(ψ::ShiftedGroupNormL0) = "shifted x ↦ Σᵢ λᵢ ‖ ‖xᵢ‖₂ ‖₀ function"
fun_expr(ψ::ShiftedGroupNormL0) = "x ↦ Σᵢ λᵢ ‖ ‖xk + sj + t‖₂"
fun_params(ψ::ShiftedGroupNormL0) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14

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
  ψ.sol .= q + ψ.xk + ψ.sj
  
  for (idx, λ) ∈ zip(ψ.h.idx, ψ.h.lambda)
    snorm = norm(ψ.sol[idx])^2
    if snorm <= 2 * γ * λ
      y[idx] .= 0
    else
      y[idx] .= ψ.sol[idx]
    end
  end
  y .-= (ψ.xk + ψ.sj)
  return y
end
