export ShiftedNormL1BInf

mutable struct ShiftedNormL1BInf{
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
  χ::Conjugate{IndBallL1{R}}
  shifted_twice::Bool

  function ShiftedNormL1BInf(
    h::NormL1{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    Δ::R,
    χ::Conjugate{IndBallL1{R}},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(sj)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, Δ, χ, shifted_twice)
  end
end

(ψ::ShiftedNormL1BInf)(y) = ψ.h(ψ.xk + ψ.sj + y) + IndBallLinf(ψ.Δ)(ψ.sj + y)

shifted(h::NormL1{R}, xk::AbstractVector{R}, Δ::R, χ::Conjugate{IndBallL1{R}}) where {R <: Real} =
  ShiftedNormL1BInf(h, xk, zero(xk), Δ, χ, false)
shifted(
  ψ::ShiftedNormL1BInf{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormL1BInf(ψ.h, ψ.xk, sj, ψ.Δ, ψ.χ, true)

fun_name(ψ::ShiftedNormL1BInf) = "shifted L1 norm with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedNormL1BInf) = "t ↦ ‖xk + sj + t‖₁ + χ({‖sj + t‖∞ ≤ Δ})"
fun_params(ψ::ShiftedNormL1BInf) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "Δ = $(ψ.Δ)"

function prox(
  ψ::ShiftedNormL1BInf{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  ψ.sol .= -ψ.sj .- ψ.xk

  for i ∈ eachindex(ψ.sol)
    ψ.sol[i] = min(max(min(max(ψ.sol[i], q[i] - ψ.λ * σ), q[i] + ψ.λ * σ), -ψ.Δ), ψ.Δ)
  end

  return ψ.sol
end
