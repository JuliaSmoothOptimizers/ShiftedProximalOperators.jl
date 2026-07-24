export ShiftedGroupNormL2Box

"""
Allows to compute the shifted GroupNormL2 operator with variable bounds: t ↦ Σᵢ λᵢ ‖xk + sj + t‖₂ + χ({sj + t .∈ [l,u]})

The proximal operator is also provided. To do so, we use the Moreau's decomposition theorem (see "Moreau’s Decomposition in Banach Spaces", P.L. Combettes et al.¹).
A direct expression of the proximal operator can be found in "Proximal Algorithms", N. Parikh et al.² 

¹https://arxiv.org/pdf/1103.3178
²https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf

"""

mutable struct ShiftedGroupNormL2Box{
  R <: Real,
  RR <: AbstractVector{R},
  I,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3,
  V4,
  VI <: AbstractArray{<:Integer},
} <: ShiftedProximableFunction
  h::GroupNormL2{R, RR, I}
  xk::V0
  sj::V1
  sol::V2
  l::V3
  u::V4
  shifted_twice::Bool
  selected::VI
  xsy::V2

  function ShiftedGroupNormL2Box(
    h::GroupNormL2{R, RR, I},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    l,
    u,
    shifted_twice::Bool,
    selected::AbstractArray{T},
  ) where {R <: Real, RR <: AbstractVector{R}, I, T <: Integer}
    sol = similar(xk)
    xsy = similar(xk, length(selected))
    if any(l .> u)
      error("Error: at least one lower bound is greater than the upper bound.")
    end
    new{R, RR, I, typeof(xk), typeof(sj), typeof(sol), typeof(l), typeof(u), typeof(selected)}(
      h,
      xk,
      sj,
      sol,
      l,
      u,
      shifted_twice,
      selected,
      xsy,
    )
  end
end

shifted(
  h::GroupNormL2{R, RR, I},
  xk::AbstractVector{R},
  l,
  u,
  selected::AbstractArray{T} = 1:length(xk),
) where {R <: Real, RR <: AbstractVector{R}, I, T <: Integer} =
  ShiftedGroupNormL2Box(h, xk, zero(xk), l, u, false, selected)

shifted(
  ψ::ShiftedGroupNormL2Box{R, RR, I, V0, V1, V2},
  sj::AbstractVector{R},
) where {
  R <: Real,
  RR <: AbstractVector{R},
  I,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} = ShiftedGroupNormL2Box(ψ.h, ψ.xk, sj, ψ.l, ψ.u, true, ψ.selected)

function (ψ::ShiftedGroupNormL2Box)(y)
  @. ψ.xsy = @views ψ.xk[ψ.selected] + ψ.sj[ψ.selected] + y[ψ.selected]
  val = ψ.h(ψ.xsy)
  ϵ = √eps(eltype(y))
  for i ∈ eachindex(y)
    lower = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
    upper = isa(ψ.u, Real) ? ψ.u : ψ.u[i]
    if !(lower - ϵ ≤ ψ.sj[i] + y[i] ≤ upper + ϵ)
      return Inf
    end
  end
  return val
end

fun_name(ψ::ShiftedGroupNormL2Box) = "shifted GroupNormL2 Σᵢ‖⋅‖₂ with box indicator"
fun_expr(ψ::ShiftedGroupNormL2Box) = "t ↦ Σᵢ λᵢ ‖xk + sj + t‖₂ + χ({sj + t .∈ [l,u]})"
fun_params(ψ::ShiftedGroupNormL2Box) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "l = $(ψ.l)\n" * " "^14 * "u = $(ψ.u)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedGroupNormL2Box{R, RR, I, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {
  R <: Real,
  RR <: AbstractVector{R},
  I,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
}
  λ = ψ.h.lambda

  @. ψ.sol = q + ψ.xk + ψ.sj
  val = zero(R)
  # Compute prox_L2 for each group without bounds
  for (idx, λ) ∈ zip(ψ.h.idx, λ)
    sol_idx = view(ψ.sol, idx)
    yv = view(y, idx)
    snorm = norm(sol_idx)
    if snorm == 0
      yv .= 0
    else
      α = max(1.0 - σ * λ / snorm, 0.0)
      @. yv = α * sol_idx
    end
  end
  @. y -= (ψ.xk + ψ.sj)
  # Apply clipping
  for i ∈ eachindex(y)
    li = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
    ui = isa(ψ.u, Real) ? ψ.u : ψ.u[i]
    si = ψ.sj[i]
    qi = q[i]
    if i ∈ ψ.selected
      y[i] = min(max(y[i], li - si), ui - si)
    else
      y[i] = prox_zero(qi, li - si, ui - si)
    end
  end
  val = zero(R)
  # Compute h(y)
  for (idx, λ) ∈ zip(ψ.h.idx, ψ.h.lambda)
    group_val = zero(R)
    for j ∈ idx
      group_val += (y[j] + ψ.xk[j] + ψ.sj[j])^2
    end
    val += λ * sqrt(group_val)
  end
  return val
end
