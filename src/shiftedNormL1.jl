export ShiftedNormL1

mutable struct ShiftedNormL1{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::NormL1{R}
  xk::V0
  sj::V1
  sol::V2
  shifted_twice::Bool

  function ShiftedNormL1(
    h::NormL1{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(xk)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice)
  end
end

shifted(h::NormL1{R}, xk::AbstractVector{R}) where {R <: Real} =
  ShiftedNormL1(h, xk, zero(xk), false)
shifted(
  ψ::ShiftedNormL1{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormL1(ψ.h, ψ.xk, sj, true)

fun_name(ψ::ShiftedNormL1) = "shifted L1 norm"
fun_expr(ψ::ShiftedNormL1) = "t ↦ ‖xk + sk + t‖₁"
fun_params(ψ::ShiftedNormL1) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)"

function prox(
  ψ::ShiftedNormL1{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  ψ.sol .= -ψ.xk .- ψ.sj

  for i ∈ eachindex(ψ.sol)
    ψ.sol[i] = min(max(ψ.sol[i], q[i] - ψ.λ * σ), q[i] + ψ.λ * σ)
  end

  return ψ.sol
end
