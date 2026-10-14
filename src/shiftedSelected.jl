export ShiftedSelected

"""
    ψ = shifted(h, xk, selected)

Return a shifted proximable function representing

    t ↦ h(xk[selected] + t[selected]),

i.e., `h` only applies to the variables in `selected`.
This works for any `h` for which `shifted(h, xk)` is defined, because

    minimize ½ σ⁻¹ ‖t - q‖² + h(xk[selected] + t[selected])

separates into the proximal operator of the shifted `h` on the `selected` indices, and `t[i] = q[i]` elsewhere.
"""
mutable struct ShiftedSelected{
  R <: Real,
  P <: ShiftedProximableFunction,
  I <: AbstractVector{<:Integer},
  V <: AbstractVector{R},
} <: ShiftedProximableFunction
  ψ::P                     # h shifted on the selected variables only
  selected::I
  unselected::Vector{Int}
end

function ShiftedSelected(
  ψ::ShiftedProximableFunction,
  x::AbstractVector{R},
  selected::AbstractVector{<:Integer},
) where {R <: Real}
  checkbounds(x, selected)
  allunique(selected) || throw(ArgumentError("selected variables must be unique"))
  unselected = setdiff(eachindex(x), selected) # compute it once instead of every time in prox!
  return ShiftedSelected(ψ, selected, unselected)
end

shifted(h, xk::AbstractVector{R}, selected::AbstractVector{<:Integer}) where {R <: Real} =
  ShiftedSelected(shifted(h, view(xk, selected)), xk, selected)

shifted(ψ::ShiftedSelected, sj::AbstractVector{R}) where {R <: Real} =
  ShiftedSelected(shifted(ψ.ψ, view(sj, ψ.selected)), sj, ψ.selected)

# Fields of the wrapper are its own; any other property, e.g., h, λ, xk or sj,
# is that of the shifted function on the selected variables.
@inline function Base.getproperty(ψ::ShiftedSelected, prop::Symbol)
  hasfield(typeof(ψ), prop) && return getfield(ψ, prop)
  return getproperty(getfield(ψ, :ψ), prop)
end

(ψ::ShiftedSelected)(y) = ψ.ψ(view(y, ψ.selected))

function shift!(ψ::ShiftedSelected, shift::AbstractVector{R}) where {R <: Real}
  shift!(ψ.ψ, view(shift, ψ.selected))
  return ψ
end

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedSelected{R},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real}
  selected, unselected = ψ.selected, ψ.unselected

  prox!(view(y, selected), ψ.ψ, view(q, selected), σ)
  view(y, unselected) .= view(q, unselected)
  return y
end

# Along the variables that are not selected, minimize ½ d y² + g y, which requires d > 0.
function iprox!(
  y::AbstractVector{R},
  ψ::ShiftedSelected{R},
  g::AbstractVector{R},
  d::AbstractVector{R},
) where {R <: Real}
  selected, unselected = ψ.selected, ψ.unselected

  iprox!(view(y, selected), ψ.ψ, view(g, selected), view(d, selected))
  view(y, unselected) .= .-view(g, unselected) ./ view(d, unselected)

  return y
end

set_radius!(ψ::ShiftedSelected, Δ::R) where {R <: Real} = throw(
  ArgumentError("use shifted(h, xk, Δ, χ, selected) for a trust region with selected variables"),
)

set_bounds!(ψ::ShiftedSelected, l, u) =
  throw(ArgumentError("use shifted(h, xk, l, u, selected) for bounds with selected variables"))

fun_name(ψ::ShiftedSelected) = fun_name(ψ.ψ) * " on selected variables"
fun_expr(ψ::ShiftedSelected) = "t ↦ h(xk[selected] + sj[selected] + t[selected])"
fun_params(ψ::ShiftedSelected) =
  "selected = $(ψ.selected)\n" * " "^14 * fun_params(ψ.ψ)
