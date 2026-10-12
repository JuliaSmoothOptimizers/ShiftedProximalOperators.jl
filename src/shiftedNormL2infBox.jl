export ShiftedNormL2InfBox

"""
    ShiftedNormL2InfBox

Shifted weighted squared ℓ₂,∞ norm with elementwise box constraint:

    ψ(t) = max_{1 ≤ i ≤ N} ωᵢ ‖(xk + sj + t)ᵢ‖²  +  χ({sj + t ∈ [l, u]})

where xk is a base shift (fixed), sj is an additional shift (updated between outer
iterations), and [l, u] is an elementwise box.
The proximal operator is computed via Propositions 2.1 & 2.2 of López-Rivera (2026),
delegating the auxiliary simplex solve to `_solve_aux_simplex` defined in NormL2Inf.jl.

# Usage
    ψ  = shifted(h, xk, l, u) # first shift  (shifted_twice = false)
    ψ2 = shifted(ψ, sj) # second shift (shifted_twice = true)
    prox!(y, ψ, q, σ) # y = prox_{σψ}(q), returns ψ(y)
"""
mutable struct ShiftedNormL2InfBox{
  R<:Real,
  V0<:AbstractVector{R},
  V1<:AbstractVector{R},
  V2<:AbstractVector{R},
  V3,
  V4,
  VI<:AbstractArray{<:Integer},
}<:ShiftedProximableFunction
  h::NormL2Inf{R}
  xk::V0
  sj::V1
  sol::V2
  l::V3 # lower bounds (scalar or vector)
  u::V4 # upper bounds (scalar or vector)
  shifted_twice::Bool
  selected::VI # indices involved in the norm (active coordinates)
  xsy::V2 # scratch buffer

  function ShiftedNormL2InfBox(
    h::NormL2Inf{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    l,
    u,
    shifted_twice::Bool,
    selected::AbstractArray{T},
  ) where {R <: Real, T <: Integer}
    any(l .> u) && error("At least one lower bound exceeds the corresponding upper bound.")
    sol = similar(xk)
    xsy = similar(xk, length(selected))
    new{R, typeof(xk), typeof(sj), typeof(sol), typeof(l), typeof(u), typeof(selected)}(
      h, xk, sj, sol, l, u, shifted_twice, selected, xsy,
    )
  end
end


"""
    shifted(h, xk, l, u, selected)

Create a `ShiftedNormL2InfBox` with base shift `xk`, bounds `[l, u]`.
"""
shifted(
  h::NormL2Inf{R},
  xk::AbstractVector{R},
  l,
  u,
  selected::AbstractArray{T} = 1:length(xk),
) where {R <: Real, T <: Integer} =
  ShiftedNormL2InfBox(h, xk, zero(xk), l, u, false, selected)

"""
    shifted(ψ, sj)

Add a secondary shift `sj` to an existing `ShiftedNormL2InfBox`.
"""
shifted(
  ψ::ShiftedNormL2InfBox{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0, V1, V2} =
  ShiftedNormL2InfBox(ψ.h, ψ.xk, sj, ψ.l, ψ.u, true, ψ.selected)


function (ψ::ShiftedNormL2InfBox{R})(y::AbstractVector{R}) where {R <: Real}
  # Feasibility check: sj + y must lie componentwise in [l, u]
  ϵ = sqrt(eps(R))
  for i ∈ eachindex(y)
    li = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
    ui = isa(ψ.u, Real) ? ψ.u : ψ.u[i]
    (li - ϵ ≤ ψ.sj[i] + y[i] ≤ ui + ϵ) || return typemax(R)
  end
  # Evaluate max_i ωᵢ ‖(xk + sj + y)ᵢ‖²
  n = _group_size(ψ.h, y)
  N = length(ψ.h.omega)
  val = zero(R)
  for i = 1:N
    rng = _group_range(i, n)
    s = zero(R)
    for j ∈ rng
      s += (ψ.xk[j] + ψ.sj[j] + y[j])^2
    end
    val = max(val, ψ.h.omega[i] * s)
  end
  return val
end


fun_name(ψ::ShiftedNormL2InfBox) = "shifted weighted squared ℓ₂,∞ norm with box indicator"
fun_expr(ψ::ShiftedNormL2InfBox) = "t ↦ maxᵢ ωᵢ ‖(xk + sj + t)ᵢ‖² + χ({sj + t ∈ [l,u]})"
fun_params(ψ::ShiftedNormL2InfBox) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" *
  " "^14 * "l = $(ψ.l)\n"  * " "^14 * "u = $(ψ.u)"


"""
    prox!(y, ψ, q, σ)

Compute  `y = prox_{σψ}(q)`  in-place for the box-constrained shifted operator.

Steps:
  1. Build v = q + xk + sj and solve the auxiliary simplex problem (Prop. 2.2)
     to obtain p ∈ ΔN.
  2. Apply Prop. 2.1: prox(v)ᵢ = vᵢ / (1 + 2 λᵢ pᵢ), subtract (xk + sj) → y.
  3. For indices in `selected`: clip y[i] to [l[i] − sj[i], u[i] − sj[i]].
  4. For indices NOT in `selected`: replace y[i] with the scalar projection
     `prox_zero(q[i], l[i] − sj[i], u[i] − sj[i])`.

Returns the value `ψ(y) = max_i ωᵢ ‖(xk + sj + y)ᵢ‖²`.
"""
function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL2InfBox{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0, V1, V2}
  h = ψ.h
  n = _group_size(h, q)
  N = length(h.omega)

  @. ψ.sol = q + ψ.xk + ψ.sj

  # αᵢ = ωᵢ ‖vᵢ‖²,   λᵢ = σ ωᵢ
  α  = Vector{R}(undef, N)
  λv = Vector{R}(undef, N)
  for i = 1:N
    vi = view(ψ.sol, _group_range(i, n))
    α[i] = h.omega[i] * sum(abs2, vi)
    λv[i] = σ * h.omega[i]
  end

  p = _solve_aux_simplex(α, λv)

  for i = 1:N
    rng = _group_range(i, n)
    vi = view(ψ.sol, rng)
    yi = view(y, rng)
    denom = one(R) + 2 * λv[i] * p[i]
    @. yi = vi / denom
  end

  @. y -= (ψ.xk + ψ.sj)

  for i ∈ eachindex(y)
    li = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
    ui = isa(ψ.u, Real) ? ψ.u : ψ.u[i]
    si = ψ.sj[i]
    if i ∈ ψ.selected
      # Clip increment so that sj[i] + y[i] ∈ [l[i], u[i]]
      y[i] = min(max(y[i], li - si), ui - si)
    else
      # Coordinate not involved in the norm: project q[i] into the box
      y[i] = prox_zero(q[i], li - si, ui - si)
    end
  end

  val = zero(R)
  for i = 1:N
    rng = _group_range(i, n)
    s = zero(R)
    for j ∈ rng
      s += (y[j] + ψ.xk[j] + ψ.sj[j])^2
    end
    val = max(val, h.omega[i] * s)
  end

  return val
end