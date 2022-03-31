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

(ψ::ShiftedRootNormLhalfBinf)(y) = ψ.h(ψ.xk + ψ.sj + y) + IndBallLinf(ψ.Δ)(ψ.sj + y)

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
  νλ = 2 * σ * ψ.λ
  ϕ(z) = acos(νλ / 8 * (abs(z) / 3)^(-3 / 2) + 0im)

  ψ.sol = q + (ψ.xk .+ ψ.sj)
  RNorm(tt, l) = 0.5 / σ * (tt - q[l])^2 + ψ.λ * sqrt(abs(tt + ψ.sj[l] + ψ.xk[l]))
  t = zeros(4) #probably not smart to use arrays, but can change
  ft = similar(t)
  for i ∈ eachindex(q)
    aqi = abs(ψ.sol[i])
    t[1] = -ψ.sj[i] - ψ.Δ
    t[2] = -ψ.sj[i] + ψ.Δ
    t[3] = 0 - ψ.xk[i] - ψ.sj[i]
    t[4] =
      real(2 * sign(ψ.sol[i]) / 3 * aqi * (1 + cos(2 * π / 3 - 2 * ϕ(ψ.sol[i]) / 3))) - ψ.xk[i] -
      ψ.sj[i]

    for j = 1:4
      if abs(t[j] + ψ.sj[i]) ≤ ψ.Δ + eps(R)
        ft[j] = RNorm(t[j], i)
      else
        ft[j] = Inf
      end
    end
    (_, a) = findmin(ft)

    y[i] = t[a]
  end

  return y
end
