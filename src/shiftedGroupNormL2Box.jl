export ShiftedGroupNormL2Box

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
    sol = similar(sj)
    xsy = similar(xk, length(selected))
    if any(l .> u)
      error("At least one lower bound is greater than the upper bound.")
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
) where {R <: Real, RR <: AbstractVector{R}, I, T <: Integer} = ShiftedGroupNormL2Box(h, xk, zero(xk), l, u, false, selected)

shifted(
  h::NormL2{R},
  xk::AbstractVector{R},
  l,
  u,
  selected::AbstractArray{T} = 1:length(xk),
) where {R <: Real, T <: Integer} = ShiftedGroupNormL2Box(GroupNormL2([h.lambda], [1:length(xk)]), xk, zero(xk), l, u, false, selected)

# Backward compatibility: Convert Binf constraints (Δ, χ) to Box constraints [-Δ, Δ]
shifted(
  h::GroupNormL2{R, RR, I},
  xk::AbstractVector{R},
  Δ::R,
  χ::Conjugate{IndBallL1{R}},
  selected::AbstractArray{T} = 1:length(xk),
) where {R <: Real, RR <: AbstractVector{R}, I, T <: Integer} = ShiftedGroupNormL2Box(h, xk, zero(xk), -Δ, Δ, false, selected)

shifted(
  h::NormL2{R},
  xk::AbstractVector{R},
  Δ::R,
  χ::Conjugate{IndBallL1{R}},
  selected::AbstractArray{T} = 1:length(xk),
) where {R <: Real, T <: Integer} = ShiftedGroupNormL2Box(GroupNormL2([h.lambda], [1:length(xk)]), xk, zero(xk), -Δ, Δ, false, selected)

shifted(
  ψ::ShiftedGroupNormL2Box{R, RR, I, V0, V1, V2, V3, V4, VI},
  sj::AbstractVector{R},
) where {R <: Real, RR <: AbstractVector{R}, I, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3, V4, VI <: AbstractArray{<:Integer}} =
  ShiftedGroupNormL2Box(ψ.h, ψ.xk, sj, ψ.l, ψ.u, true, ψ.selected)

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

fun_name(ψ::ShiftedGroupNormL2Box) = "shifted ∑ᵢ‖⋅‖₂ norm with box indicator"
fun_expr(ψ::ShiftedGroupNormL2Box) = "t ↦ ∑ᵢ ‖xk + sj + t‖₂ + χ({sj + t .∈ [l,u]})"
fun_params(ψ::ShiftedGroupNormL2Box) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "lb = $(ψ.l)\n" * " "^14 * "ub = $(ψ.u)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedGroupNormL2Box{R, RR, I, V0, V1, V2, V3, V4, VI},
  q::AbstractVector{R},
  σ::R,
) where {
  R <: Real,
  RR <: AbstractVector{R},
  I,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3,
  V4,
  VI <: AbstractArray{<:Integer},
}
  ψ.sol .= q .+ ψ.xk .+ ψ.sj

  # buffer to reuse for block computations
  tmp = similar(ψ.sol)

  for (idx, λ) ∈ zip(ψ.h.idx, ψ.h.lambda)
    σλ = λ * σ
    @views begin
      solb = ψ.sol[idx]
      xkb = ψ.xk[idx]
      sjb = ψ.sj[idx]
    end

    # compute tmpb = solb .- xkb .- sjb
    tmpb = tmp[1:length(solb)]
    @inbounds for i in eachindex(solb)
      tmpb[i] = solb[i] - xkb[i] - sjb[i]
    end

    # l2prox in-place into tmpb
    s = zero(eltype(tmpb))
    @inbounds for i in eachindex(tmpb)
      s += tmpb[i]^2
    end
    s = sqrt(s)
    factor = s == 0 ? zero(eltype(s)) : max(0, 1 - σλ / s)
    @inbounds for i in eachindex(tmpb)
      tmpb[i] = factor * tmpb[i]
    end

    # Apply box constraints elementwise and write to y
    @inbounds for (i, global_i) in enumerate(idx)
      li = isa(ψ.l, Real) ? ψ.l : ψ.l[global_i]
      ui = isa(ψ.u, Real) ? ψ.u : ψ.u[global_i]
      y[global_i] = min(max(tmpb[i], li), ui)
    end
  end
  return y
end