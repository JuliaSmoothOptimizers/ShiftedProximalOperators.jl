export ShiftedNormLinfBox

mutable struct ShiftedNormLinfBox{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3,
  V4,
  VI <: AbstractArray{<:Integer},
} <: ShiftedProximableFunction
  h::Conjugate{IndBallL1{R}}
  xk::V0
  sj::V1
  sol::V2
  l::V3
  u::V4
  shifted_twice::Bool
  selected::VI
  xsy::V2

  function ShiftedNormLinfBox(
    h::Conjugate{IndBallL1{R}},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    l,
    u,
    shifted_twice::Bool,
    selected::AbstractArray{T},
  ) where {R <: Real, T <: Integer}
    sol = similar(xk)
    xsy = similar(xk, length(selected))
    if any(l .> u)
      error("Error: at least one lower bound is greater than the upper bound.")
    end
    new{R, typeof(xk), typeof(sj), typeof(sol), typeof(l), typeof(u), typeof(selected)}(
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
  h::Conjugate{IndBallL1{R}},
  xk::AbstractVector{R},
  l,
  u,
  selected::AbstractArray{T} = 1:length(xk),
) where {R <: Real, T <: Integer} =
  ShiftedNormLinfBox(h, xk, zero(xk), l, u, false, selected)

shifted(
  ψ::ShiftedNormLinfBox{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormLinfBox(ψ.h, ψ.xk, sj, ψ.l, ψ.u, true, ψ.selected)
shifted(
  ψ::ShiftedNormLinfBox{R, V0, V1, V2},
  sj::AbstractVector{R},
  l,
  u,
  selected::AbstractArray{T} = 1:length(sj),
) where {R <: Real, T <: Integer, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormL1Box(ψ.h, ψ.xk, sj, l, u, true, selected)

function (ψ::ShiftedNormLinfBox)(y)
  tmp = ψ.xk .+ ψ.sj .+ y
  val = ψ.h(tmp)
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

fun_name(ψ::ShiftedNormLinfBox) = "shifted L∞ norm with box indicator"
fun_expr(ψ::ShiftedNormLinfBox) = "t ↦ λ ‖xk + sj + t‖∞ + χ({sj + t .∈ [l,u]})"
fun_params(ψ::ShiftedNormLinfBox) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "l = $(ψ.l)\n" * " "^14 * "u = $(ψ.u)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormLinfBox{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  λ = ψ.h.f.r
  @. ψ.sol = q + ψ.xk + ψ.sj

  r = σ * λ
  y .= ψ.sol .- _proj_l1ball(ψ.sol, r)
  # @. y -= (ψ.xk + ψ.sj)

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
  @inbounds for i ∈ ψ.selected
    val = max(val, λ * abs(y[i] + ψ.xk[i] + ψ.sj[i]))
  end
  return val
end