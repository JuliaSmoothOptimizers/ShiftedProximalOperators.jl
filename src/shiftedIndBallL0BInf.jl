export ShiftedIndBallL0BInf

mutable struct ShiftedIndBallL0BInf{
  I <: Integer,
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::IndBallL0{I}
  x0::V0
  x::V1
  s::V2
  p::Vector{Int}
  Δ::R
  χ::Conjugate{IndBallL1{R}}
  function ShiftedIndBallL0BInf(
    h::IndBallL0{I},
    x0::AbstractArray{R},
    x::AbstractVector{R},
    Δ::R,
    χ::Conjugate{IndBallL1{R}},
  ) where {I <: Integer, R <: Real}
    s = similar(x)
    new{I, R, typeof(x0), typeof(x), typeof(s)}(h, x0, x, s, Vector{Int}(undef, length(x)), Δ, χ)
  end
end

(ψ::ShiftedIndBallL0BInf)(y) = ψ.h(ψ.x0 + ψ.x + y) + IndBallLinf(ψ.Δ)(y)

shifted(
  h::IndBallL0{I},
  x::AbstractVector{R},
  Δ::R,
  χ::Conjugate{IndBallL1{R}},
) where {I <: Integer, R <: Real} = ShiftedIndBallL0BInf(h, zero(x), x, Δ, χ)
shifted(
  ψ::ShiftedIndBallL0BInf{I, R, V0, V1, V2},
  x::AbstractVector{R},
) where {
  I <: Integer,
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} = ShiftedIndBallL0BInf(ψ.h, ψ.x, x, ψ.Δ, ψ.χ)

fun_name(ψ::ShiftedIndBallL0BInf) = "shifted L0 norm ball with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedIndBallL0BInf) = "s ↦ χ({‖x + s‖₀ ≤ r}) + χ({‖s‖∞ ≤ Δ})"
fun_params(ψ::ShiftedIndBallL0BInf) = "x0 = $(ψ.x0)\n" * " "^14 * "x = $(ψ.x), Δ = $(ψ.Δ)"

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
  ψ.s .= ψ.x .+ ψ.x0 .+ q
  # find largest entries
  sortperm!(ψ.p, ψ.s, rev = true, by = abs) # stock with ψ.p as placeholder
  ψ.s[ψ.p[(ψ.h.r + 1):end]] .= 0 # set smallest to zero

  for i ∈ eachindex(ψ.s)
    ψ.s[i] = min(max(ψ.s[i], ψ.x0[i] + ψ.x[i] - ψ.Δ), ψ.x0[i] + ψ.x[i] + ψ.Δ)
  end

  ψ.s .-= ψ.x .+ ψ.x0
  return ψ.s
end
