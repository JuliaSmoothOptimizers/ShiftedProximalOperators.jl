export ShiftedNormL2Inf

"""
    ShiftedNormL2Inf

Shifted weighted squared ℓ₂,∞ norm:

    ψ(t) = max_{1 ≤ i ≤ N} ωᵢ ‖(xk + sj + t)ᵢ‖²

where xk is a base shift (fixed) and sj is an additional shift (updated between outer iterations).
The proximal operator is computed via Propositions 2.1 & 2.2 of López-Rivera (2026),
delegating the auxiliary simplex solve to `_solve_aux_simplex` defined in NormL2Inf.jl.

# Usage
    ψ = shifted(h, xk) # first shift  (shifted_twice = false)
    ψ2 = shifted(ψ, sj) # second shift (shifted_twice = true)
    prox!(y, ψ, q, σ) # y = prox_{σψ}(q), returns ψ(y)
"""
mutable struct ShiftedNormL2Inf{
  R<:Real,
  V0<:AbstractVector{R},
  V1<:AbstractVector{R},
  V2<:AbstractVector{R},
}<:ShiftedProximableFunction
  h::NormL2Inf{R}
  xk::V0
  sj::V1
  sol::V2 # full shifted argument q + xk + sj
  shifted_twice::Bool
  xsy::V2 # buffer (same length as sol)
  α::Vector{R} # αᵢ
  λv::Vector{R} # λᵢ
  p::Vector{R} # solution simplex
  order::Vector{Int} # buffer
  α_sqrt::Vector{R} # buffer
  λ_sqrt::Vector{R} # buffer

  function ShiftedNormL2Inf(
    h::NormL2Inf{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(sj)
    xsy = similar(sj)
    N = length(h.omega)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(
      h, xk, sj, sol, shifted_twice, xsy,
      Vector{R}(undef, N), Vector{R}(undef, N), Vector{R}(undef, N),
      Vector{Int}(undef, N), Vector{R}(undef, N), Vector{R}(undef, N),
    )
  end
end

"""
    shifted(h, xk)

Create a `ShiftedNormL2Inf` with base shift `xk` and zero secondary shift.
"""
shifted(h::NormL2Inf{R}, xk::AbstractVector{R}) where {R <: Real} =
  ShiftedNormL2Inf(h, xk, zero(xk), false)

"""
    shifted(ψ, sj)

Add a secondary shift `sj` to an existing `ShiftedNormL2Inf`.
"""
shifted(
  ψ::ShiftedNormL2Inf{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0, V1, V2} =
  ShiftedNormL2Inf(ψ.h, ψ.xk, sj, true)


fun_name(ψ::ShiftedNormL2Inf) = "shifted weighted squared ℓ₂,∞ norm"
fun_expr(ψ::ShiftedNormL2Inf) = "t ↦ maxᵢ ωᵢ ‖(xk + sj + t)ᵢ‖²"
fun_params(ψ::ShiftedNormL2Inf) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n"


"""
    prox!(y, ψ, q, σ)

Compute `y = prox_{σψ}(q)`, where `ψ(t) = h(xk + sj + t)`.
"""
function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL2Inf{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0, V1, V2}
  h = ψ.h
  n = _group_size(h, q)
  N = length(h.omega)

  @. ψ.sol = q + ψ.xk + ψ.sj

  # αᵢ = ωᵢ ‖vᵢ‖², λᵢ = σ ωᵢ
  for i = 1:N
    vi = view(ψ.sol, _group_range(i, n))
    ψ.α[i] = h.omega[i] * sum(abs2, vi)
    ψ.λv[i] = σ * h.omega[i]
  end

  # Solve auxiliary simplex problem → p ∈ ΔN  (Prop. 2.2)
  _solve_aux_simplex!(ψ.p, ψ.α, ψ.λv, ψ.order, ψ.α_sqrt, ψ.λ_sqrt)

  # Apply Prop. 2.1:  prox_{σh}(v)ᵢ = vᵢ / (1 + 2 λᵢ pᵢ),
  # then subtract (xk + sj) to get the increment y
  val = zero(R)
  for i = 1:N
    rng = _group_range(i, n)
    vi = view(ψ.sol, rng)
    yi = view(y, rng)
    denom = one(R) + 2 * ψ.λv[i] * ψ.p[i]
    @. yi = vi / denom
    val = max(val, h.omega[i] * sum(abs2, yi))
  end

  # Convert absolute prox value to increment
  @. y -= (ψ.xk + ψ.sj)

  return val
end