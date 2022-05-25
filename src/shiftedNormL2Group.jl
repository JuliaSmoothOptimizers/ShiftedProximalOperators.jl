export ShiftedNormL2Group

mutable struct ShiftedNormL2Group{
  R <: Real,
  RR <: AbstractVector{R},
  I <: Vector{Vector{Int}},
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::GroupNormL2{R, RR, I}
  xk::V0
  sj::V1
  sol::V2
  shifted_twice::Bool

  function ShiftedNormL2Group(
    h::GroupNormL2{R, RR, I},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real, RR <:  AbstractVector{R}, I <: Vector{Vector{Int}}}
    sol = similar(sj)
    new{R, RR, I, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice)
  end
end

shifted(
  h::GroupNormL2{R,RR, I},
  xk::AbstractVector{R},
) where {R <: Real, RR <: AbstractVector{R}, I <: Vector{Vector{Int}}} = ShiftedNormL2Group(h, xk, zero(xk), false)
shifted(
  ψ::ShiftedNormL2Group{R, RR, I, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, RR <:  AbstractVector{R}, I <: Vector{Vector{Int}}, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
ShiftedNormL2Group(ψ.h, ψ.xk, sj, true)

fun_name(ψ::ShiftedNormL2Group) = "shifted ∑ᵢ||⋅||₂ norm"
fun_expr(ψ::ShiftedNormL2Group) = "t ↦ ∑ᵢ ‖xk + sj + t‖₂"
fun_params(ψ::ShiftedNormL2Group) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL2Group{R, RR, I, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, RR <: AbstractVector{R}, I <: Vector{Vector{Int}}, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}

  ψ.sol .= q + ψ.xk + ψ.sj
  if length(ψ.h.idx) == 1
    snorm = sqrt(sum(ψ.sol.^2))
    y .= max(1 - σ*ψ.h.lambda/snorm, 0) .* ψ.sol
  else
    for i = 1:length(ψ.h.idx)
      snorm = sqrt(sum(ψ.sol[ψ.h.idx[i]].^2))
      y[ψ.h.idx[i]] .= max(1 - σ*ψ.h.lambda[i]/snorm, 0) .* ψ.sol[ψ.h.idx[i]]
    end
  end
  y .-= (ψ.xk + ψ.sj)
  return y
end
