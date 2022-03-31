export ShiftedRootNormLhalf
using LinearAlgebra

mutable struct ShiftedRootNormLhalf{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::RootNormLhalf{R}
  xk::V0
  sj::V1
  sol::V2
  shifted_twice::Bool

  function ShiftedRootNormLhalf(
    h::RootNormLhalf{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(xk)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice)
  end
end

shifted(h::RootNormLhalf{R}, xk::AbstractVector{R}) where {R <: Real} =
  ShiftedRootNormLhalf(h, xk, zero(xk), false)
shifted(
  ψ::ShiftedRootNormLhalf{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedRootNormLhalf(ψ.h, ψ.xk, sj, true)

fun_name(ψ::ShiftedRootNormLhalf) = "shifted L½  norm"
fun_expr(ψ::ShiftedRootNormLhalf) = "t ↦ ‖xk + sk + t‖ₚᵖ, p = 1/2"
fun_params(ψ::ShiftedRootNormLhalf) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedRootNormLhalf{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  νλ = σ * ψ.λ
  ϕ(z) = acos(νλ / 4 * (abs(z) / 3)^(-3 / 2))
  p = 54^(1 / 3) * (2νλ)^(2 / 3) / 4
  ψ.sol = q + (ψ.xk .+ ψ.sj)

  for i ∈ eachindex(y)
    aqi = abs(ψ.sol[i])
    if aqi ≤ p
      y[i] = 0
    else
      y[i] = 2 * sign(ψ.sol[i]) / 3 * aqi * (1 + cos(2 * π / 3 - 2 * ϕ(ψ.sol[i]) / 3))
    end
    y[i] -= (ψ.xk[i] + ψ.sj[i])
  end

  return y
end
