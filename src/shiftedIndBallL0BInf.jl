export ShiftedIndBallL0BInf

mutable struct ShiftedIndBallL0BInf{
  I <: Integer,
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::IndBallL0{I}
  xk::V0
  sj::V1
  sol::V2
  p::Vector{Int}
  Δ::R
  χ::Conjugate{IndBallL1{R}}
  shifted_twice::Bool
  function ShiftedIndBallL0BInf(
    h::IndBallL0{I},
    xk::AbstractArray{R},
    sj::AbstractVector{R},
    Δ::R,
    χ::Conjugate{IndBallL1{R}},
    shifted_twice::Bool,
  ) where {I <: Integer, R <: Real}
    sol = similar(sj)
    new{I, R, typeof(xk), typeof(sj), typeof(sol)}(
      h,
      xk,
      sj,
      sol,
      Vector{Int}(undef, length(sj)),
      Δ,
      χ,
      shifted_twice,
    )
  end
end

# TODO: find a more robust solution than the factor 1.1 here.
(ψ::ShiftedIndBallL0BInf)(y) = ψ.h(ψ.xk + ψ.sj + y) + IndBallLinf(1.1 * ψ.Δ)(ψ.sj + y)

shifted(
  h::IndBallL0{I},
  xk::AbstractVector{R},
  Δ::R,
  χ::Conjugate{IndBallL1{R}},
) where {I <: Integer, R <: Real} = ShiftedIndBallL0BInf(h, xk, zero(xk), Δ, χ, false)
shifted(
  ψ::ShiftedIndBallL0BInf{I, R, V0, V1, V2},
  sj::AbstractVector{R},
) where {
  I <: Integer,
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} = ShiftedIndBallL0BInf(ψ.h, ψ.xk, sj, ψ.Δ, ψ.χ, true)

fun_name(ψ::ShiftedIndBallL0BInf) = "shifted L0 norm ball with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedIndBallL0BInf) = "t ↦ χ({‖xk + sj + t‖₀ ≤ r}) + χ({‖sj + t‖∞ ≤ Δ})"
fun_params(ψ::ShiftedIndBallL0BInf) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)," * " "^14 * "Δ = $(ψ.Δ)"

function prox(
  ψ::ShiftedIndBallL0BInf{I, R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {
  I <: Integer,
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
}
  ψ.sol .= ψ.xk .+ ψ.sj .+ q
  # find largest entries
  sortperm!(ψ.p, ψ.sol, rev = true, by = abs) # stock with ψ.p as placeholder
  ψ.sol[ψ.p[(ψ.h.r + 1):end]] .= 0 # set smallest to zero

  for i ∈ eachindex(ψ.sol)
    ψ.sol[i] = min(max(ψ.sol[i] - (ψ.xk[i] + ψ.sj[i]), -ψ.Δ), ψ.Δ)
  end

  return ψ.sol
end
