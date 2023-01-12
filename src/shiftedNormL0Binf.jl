export ShiftedNormL0BInf

mutable struct ShiftedNormL0BInf{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::NormL0{R}
  xk::V0
  sj::V1
  sol::V2
  Δ::R
  χ::Conjugate{IndBallL1{R}}
  shifted_twice::Bool

  function ShiftedNormL0BInf(
    h::NormL0{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    Δ::R,
    χ::Conjugate{IndBallL1{R}},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(xk)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, Δ, χ, shifted_twice)
  end
end

function (ψ::ShiftedNormL0BInf)(y)
  return ψ.h(ψ.xk + ψ.sj + y) + IndBallLinf(1.01 * ψ.Δ)(ψ.sj + y)
end

shifted(h::NormL0{R}, xk::AbstractVector{R}, Δ::R, χ::Conjugate{IndBallL1{R}}) where {R <: Real} =
  ShiftedNormL0BInf(h, xk, zero(xk), Δ, χ, false)
shifted(
  ψ::ShiftedNormL0BInf{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormL0BInf(ψ.h, ψ.xk, sj, ψ.Δ, ψ.χ, true)

fun_name(ψ::ShiftedNormL0BInf) = "shifted L0 pseudo-norm with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedNormL0BInf) = "t ↦ λ ‖xk + sj + t‖₀ + χ({‖sj + t‖∞ ≤ Δ})"
fun_params(ψ::ShiftedNormL0BInf) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "Δ = $(ψ.Δ)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL0BInf{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  c2 = 2 * ψ.λ * σ
  c = sqrt(c2)

  for i ∈ eachindex(q)
    xs = ψ.xk[i] + ψ.sj[i]
    xsq = xs + q[i]
    left = ψ.xk[i] - ψ.Δ
    right = ψ.xk[i] + ψ.Δ
    val_left = (left - xsq)^2 + (ψ.xk[i] == ψ.Δ ? 0 : c2)
    val_right = (right - xsq)^2 + (ψ.xk[i] == -ψ.Δ ? 0 : c2)
    # subtract x + s from solution explicitly here instead of doing it
    # numerically at the end
    y[i] = val_left < val_right ? (-ψ.sj[i] - ψ.Δ) : (-ψ.sj[i] + ψ.Δ)
    val_min = min(val_left, val_right)
    val_0 = xsq^2
    val_xsq = xsq == 0 ? zero(R) : c2
    if left ≤ 0 ≤ right
      val_0 < val_min && (y[i] = -xs)
      val_min = min(val_0, val_min)
    end
    if left ≤ xsq ≤ right
      val_xsq < val_min && (y[i] = q[i])
    end
  end

  return y
end
