export ShiftedNormL2

mutable struct ShiftedNormL2{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::NormL2{R}
  xk::V0
  sj::V1
  sol::V2
  shifted_twice::Bool

  function ShiftedNormL2(
    h::RootNormLhalf{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(sj)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice)
  end
end

shifted(h::NormL2{R}, xk::AbstractVector{R}) where {R <: Real} = 
  ShiftedNormL2(h, xk, zero(xk), false)
shifted(
  ψ::ShiftedNormL2{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormL2(ψ.h, ψ.xk, sj, true)

fun_name(ψ::ShiftedNormL2Group) = "shifted L1 norm"
fun_expr(ψ::ShiftedNormL2Group) = "t ↦ ‖xk + sj + t‖\_2, p = 2 - fix"
fun_params(ψ::ShiftedNormL2Group) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL2{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}

  ψ.sol .= q + ψ.xk + ψ.sj
  y .= max(1 - σ/norm(ψ.sol), 0) .* ψ.sol - (ψ.xk + ψ.sj)

  return y
end
