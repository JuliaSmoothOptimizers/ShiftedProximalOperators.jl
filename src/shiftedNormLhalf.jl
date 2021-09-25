export ShiftedNormLhalf

mutable struct ShiftedNormLhalf{
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

  function ShiftedNormLhalf(
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
  ShiftedNormLhalf(h, xk, zero(xk), false)
shifted(
  ψ::ShiftedNormLhalf{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormLhalf(ψ.h, ψ.xk, sj, true)

fun_name(ψ::ShiftedNormLhalf) = "shifted Lhalf norm"
fun_expr(ψ::ShiftedNormLhalf) = "t ↦ ‖xk + sk + t‖ₚᵖ, p = 1/2"
fun_params(ψ::ShiftedNormLhalf) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormLhalf{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  γλ = 2 * σ * ψ.λ
  ϕ(z) = acos(γλ / 8 * (abs(z) /3 )^(-3/2))
  p = 54^(1/3) * (γλ^(2/3)) / 4

  q .+= (ψ.xk .+ ψ.sj)

  for i ∈ eachindex(y)
    aqi = abs(q[i])
    if aqi ≤ p
      y[i] = 0
    else
      y[i] = 2 * sign(q[i]) / 3 * aqi * (1 + cos(2 * π / 3 - 2 * ϕ(q[i]) / 3))
    end
    y[i] -= (ψ.xk[i] + ψ.sj[i])
  end

  return y
end
