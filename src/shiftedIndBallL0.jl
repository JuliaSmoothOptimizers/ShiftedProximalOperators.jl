export ShiftedIndBallL0

mutable struct ShiftedIndBallL0{
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
  shifted_twice::Bool

  function ShiftedIndBallL0(
    h::IndBallL0{I},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {I <: Integer, R <: Real}
    sol = similar(sj)
    new{I, R, typeof(xk), typeof(sj), typeof(sol)}(
      h,
      xk,
      sj,
      sol,
      Vector{Int}(undef, length(sj)),
      shifted_twice,
    )
  end
end

shifted(h::IndBallL0{I}, xk::AbstractVector{R}) where {I <: Integer, R <: Real} =
  ShiftedIndBallL0(h, xk, zero(xk), false)
shifted(
  ψ::ShiftedIndBallL0{I, R, V0, V1, V2},
  sj::AbstractVector{R},
) where {
  I <: Integer,
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} = ShiftedIndBallL0(ψ.h, ψ.xk, sj, true)

fun_name(ψ::ShiftedIndBallL0) = "shifted L0 norm ball indicator"
fun_expr(ψ::ShiftedIndBallL0) = "t ↦ χ({‖xk + sj + t‖₀ ≤ r})"

function prox(
  ψ::ShiftedIndBallL0{I, R, V0, V1, V2},
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
  ψ.sol .-= ψ.xk .+ ψ.sj
  return ψ.sol
end
