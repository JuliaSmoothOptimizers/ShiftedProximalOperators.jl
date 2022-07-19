export ShiftedRootNormLhalfBinf

mutable struct ShiftedRootNormLhalfBinf{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::RootNormLhalf{R}
  xk::V0
  sj::V1
  sol::V2
  Δ::R
  χ::Conjugate{IndBallL1{R}}
  shifted_twice::Bool

  function ShiftedRootNormLhalfBinf(
    h::RootNormLhalf{R},
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

(ψ::ShiftedRootNormLhalfBinf)(y) =
  ψ.h(ψ.xk + ψ.sj + y) +
  (ψ.χ(ψ.sj + y) ≈ ψ.Δ ? IndBallLinf(ψ.Δ)([ψ.Δ]) : IndBallLinf(ψ.Δ)(ψ.sj + y))

shifted(
  h::RootNormLhalf{R},
  xk::AbstractVector{R},
  Δ::R,
  χ::Conjugate{IndBallL1{R}},
) where {R <: Real} = ShiftedRootNormLhalfBinf(h, xk, zero(xk), Δ, χ, false)
shifted(
  ψ::ShiftedRootNormLhalfBinf{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedRootNormLhalfBinf(ψ.h, ψ.xk, sj, ψ.Δ, ψ.χ, true)

fun_name(ψ::ShiftedRootNormLhalfBinf) = "shifted ∑ᵢ√|⋅| norm with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedRootNormLhalfBinf) = "t ↦ ‖xk + sj + t‖ₚᵖ + χ({‖sj + t‖∞ ≤ Δ}), p = 1/2"
fun_params(ψ::ShiftedRootNormLhalfBinf) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "Δ = $(ψ.Δ)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedRootNormLhalfBinf{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  ϕ(z) = acos(σ * ψ.λ / 4 * (abs(z) / 3)^(-3 / 2) + 0im)

  ψ.sol .= (ψ.xk .+ ψ.sj)
  RNorm(tt, l) = 0.5 / σ * (tt - q[l])^2 + ψ.λ * sqrt(abs(tt + ψ.sol[l]))
  for i ∈ eachindex(q)
    aqi = abs(ψ.sol[i] + q[i])
    val =
      real(2 * sign(ψ.sol[i] + q[i]) / 3 * aqi * (1 + cos(2 * π / 3 - 2 * ϕ(ψ.sol[i] + q[i]) / 3))) #+ t[3]

    (_, a) = findmin((
      RNorm(-ψ.sj[i] - ψ.Δ, i),
      RNorm(-ψ.sj[i] + ψ.Δ, i),
      abs(-ψ.xk[i]) ≤ ψ.Δ + eps(R) ? RNorm(-ψ.sol[i], i) : Inf,
      abs(val - ψ.xk[i]) + eps(R) ≤ ψ.Δ ? RNorm(val - ψ.sol[i], i) : Inf,
    ))
    y[i] = a == 1 ? (-ψ.sj[i] - ψ.Δ) : a == 2 ? -ψ.sj[i] + ψ.Δ : a == 3 ? -ψ.sol[i] : val - ψ.sol[i]
  end

  return y
end
