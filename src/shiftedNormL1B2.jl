export ShiftedNormL1B2

mutable struct ShiftedNormL1B2{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::NormL1{R}
  xk::V0
  sj::V1
  sol::V2
  Δ::R
  χ::NormL2{R}
  shifted_twice::Bool

  function ShiftedNormL1B2(
    h::NormL1{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    Δ::R,
    χ::NormL2{R},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(sj)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, Δ, χ, shifted_twice)
  end
end

(ψ::ShiftedNormL1B2)(y) = ψ.h(ψ.xk + ψ.sj + y) + IndBallL2(ψ.Δ)(ψ.sj + y)

shifted(h::NormL1{R}, xk::AbstractVector{R}, Δ::R, χ::NormL2{R}) where {R <: Real} =
  ShiftedNormL1B2(h, xk, zero(xk), Δ, χ, false)
shifted(
  ψ::ShiftedNormL1B2{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormL1B2(ψ.h, ψ.xk, sj, ψ.Δ, ψ.χ, true)

fun_name(ψ::ShiftedNormL1B2) = "shifted L1 norm with L2-norm trust region indicator"
fun_expr(ψ::ShiftedNormL1B2) = "t ↦ ‖xk + sj + t‖₁ + χ({‖sj + t‖₂ ≤ Δ})"
fun_params(ψ::ShiftedNormL1B2) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "Δ = $(ψ.Δ)"

function prox(
  ψ::ShiftedNormL1B2{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  ProjB(y) = min.(max.(y, ψ.sj .+ q .- ψ.λ * σ), ψ.sj .+ q .+ ψ.λ * σ)
  froot(η) = η - ψ.χ(ProjB((- ψ.xk) .* (η / ψ.Δ)))

  ψ.sol .= ProjB(- ψ.xk)

  if ψ.Δ ≤ ψ.χ(ψ.sol)
    η = fzero(froot, 1e-10, Inf)
    ψ.sol .= ProjB((- ψ.xk) .* (η / ψ.Δ)) * (ψ.Δ / η)
  end
  ψ.sol .-= ψ.sj
  return ψ.sol
end
