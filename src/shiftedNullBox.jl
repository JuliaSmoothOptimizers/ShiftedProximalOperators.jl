# Null box regularizer
export ShiftedNullRegularizerBox

@doc raw"""
    ShiftedNullRegularizerBox(h, sj, shifted_twice, l, u)

Returns the shifted box null regularizer, i.e., the function that is identically zero on a box and +∞ outside of it.
```math
ψ(x) = h(xk + sj + x) + \chi(sj + x | [l, u])
```
where `h` is identically zero, `xk`represents a shift, `sj` is an additional shift that is applied to the indicator 
function as well and `[l, u]` is the box that defines the domain of the function.

### Arguments
- `h`: The unshifted null regularizer (see `NullRegularizer`).
- `sj`: The shift of the indicator function.
- `shifted_twice`: A boolean indicating whether `sj` is updated or not on shifts, true means that `sj` is updated, false means that `xk` is updated.
- `l`: The lower bound of the box.
- `u`: The upper bound of the box.
"""
mutable struct ShiftedNullRegularizerBox{T <: Real, V <: AbstractVector{T}, VT <: Union{AbstractVector{T}, T}} <: ShiftedProximableFunction 
  h::NullRegularizer{T}
  sj::V
  shifted_twice::Bool
  l::VT
  u::VT

  function ShiftedNullRegularizerBox(
    h::NullRegularizer{T},
    sj::V,
    shifted_twice::Bool,
    l::VT,
    u::VT,
  ) where {T <: Real, V <: AbstractVector{T}, VT <: Union{AbstractVector{T}, T}}
    new{T, V, VT}(h, sj, shifted_twice, l, u)
  end
end

shifted(
  h::NullRegularizer{T},
  xk::AbstractVector{T},
  l,
  u,
  selected::AbstractArray{I} = 1:length(xk),
) where {T <: Real, I <: Integer} = shifted(h, xk, l, u)
shifted(
  h::NullRegularizer{T},
  xk::AbstractVector{T},
  l::VT,
  u::VT,
) where {T <: Real, VT} = ShiftedNullRegularizerBox(h, zero(xk), false, l, u)
shifted(
  h::NullRegularizer{T},
  xk::AbstractVector{T},
  Δ::T,
  χ::Conjugate{IndBallL1{T}},
) where {T <: Real} = ShiftedNullRegularizerBox(h, zero(xk), false, -Δ, Δ)
shifted(
  ψ::ShiftedNullRegularizerBox{T, V, VT},
  sj::AbstractVector{T},
) where {T <: Real, V <: AbstractVector{T}, VT} = ShiftedNullRegularizerBox(ψ.h, sj, true, ψ.l, ψ.u)

function shift!(ψ::ShiftedNullRegularizerBox{T, V, VT}, shift::AbstractVector{T}) where {T <: Real, V <: AbstractVector{T}, VT}
  ψ.shifted_twice && (ψ.sj .= shift)
end

function (ψ::ShiftedNullRegularizerBox{T, V})(y) where {T <: Real, V <: AbstractVector{T}}
  ϵ = √eps(eltype(y))
  @inbounds for i in eachindex(y)
    l = ψ.l isa AbstractVector ? ψ.l[i] : ψ.l
    u = ψ.u isa AbstractVector ? ψ.u[i] : ψ.u
    if !(l - ϵ ≤ ψ.sj[i] + y[i] ≤ u + ϵ)
      return T(Inf)
    end
  end
  return zero(T)
end

fun_name(ψ::ShiftedNullRegularizerBox{T, V}) where {T <: Real, V <: AbstractVector{T}} = "shifted null regularizer with box indicator"
fun_expr(ψ::ShiftedNullRegularizerBox{T, V}) where {T <: Real, V <: AbstractVector{T}} = "t ↦ χ({sj + t .∈ [l,u]})"
fun_params(ψ::ShiftedNullRegularizerBox{T, V}) where {T <: Real, V <: AbstractVector{T}} =
  "sj = $(ψ.sj)\n" * " "^14 * "l = $(ψ.l)\n" * " "^14 * "u = $(ψ.u)"

function prox!(
  y::AbstractVector{T},
  ψ::ShiftedNullRegularizerBox{T, V},
  q::AbstractVector{T},
  σ::T,
) where {T <: Real, V <: AbstractVector{T}}
  @assert σ > zero(T)
  @inbounds for i ∈ eachindex(y)
    l = ψ.l isa AbstractVector ? ψ.l[i] : ψ.l
    u = ψ.u isa AbstractVector ? ψ.u[i] : ψ.u
    y[i] = prox_zero(q[i], l - ψ.sj[i], u - ψ.sj[i])
  end
  return y
end

function iprox!(
  y::AbstractVector{T},
  ψ::ShiftedNullRegularizerBox{T, V},
  g::AbstractVector{T},
  d::AbstractVector{T},
) where {T <: Real, V <: AbstractVector{T}}
  @inbounds for i ∈ eachindex(y)
    l = ψ.l isa AbstractVector ? ψ.l[i] : ψ.l
    u = ψ.u isa AbstractVector ? ψ.u[i] : ψ.u
    y[i] = iprox_zero(d[i], g[i], l - ψ.sj[i], u - ψ.sj[i])
  end
  return y
end