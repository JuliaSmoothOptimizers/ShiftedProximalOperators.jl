export ShiftedIndBallL0Box

mutable struct ShiftedIndBallL0Box{
  I <: Integer,
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3,
  V4,
  VI <: AbstractArray{<:Integer},
} <: ShiftedProximableFunction
  h::IndBallL0{I}
  xk::V0
  sj::V1
  sol::V2
  p::Vector{Int}
  l::V3
  u::V4
  shifted_twice::Bool
  selected::VI
  xsy::V2

  function ShiftedIndBallL0Box(
    h::IndBallL0{I},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    l,
    u,
    shifted_twice::Bool,
    selected::AbstractArray{T},
  ) where {I <: Integer, R <: Real, T <: Integer}
    sol = similar(sj)
    xsy = similar(xk, length(selected))
    if any(l .> u)
      error("At least one lower bound is greater than the upper bound.")
    end
    new{I, R, typeof(xk), typeof(sj), typeof(sol), typeof(l), typeof(u), typeof(selected)}(
      h,
      xk,
      sj,
      sol,
      Vector{Int}(undef, length(sj)),
      l,
      u,
      shifted_twice,
      selected,
      xsy,
    )
  end
end

shifted(
  h::IndBallL0{I},
  xk::AbstractVector{R},
  l,
  u,
  selected::AbstractArray{T} = 1:length(xk),
) where {I <: Integer, R <: Real, T <: Integer} = ShiftedIndBallL0Box(h, xk, zero(xk), l, u, false, selected)

# Backward compatibility: Convert Binf constraints (Δ, χ) to Box constraints [-Δ, Δ]
shifted(
  h::IndBallL0{I},
  xk::AbstractVector{R},
  Δ::R,
  χ::Conjugate{IndBallL1{R}},
  selected::AbstractArray{T} = 1:length(xk),
) where {I <: Integer, R <: Real, T <: Integer} = ShiftedIndBallL0Box(h, xk, zero(xk), -Δ, Δ, false, selected)

shifted(
  ψ::ShiftedIndBallL0Box{I, R, V0, V1, V2, V3, V4, VI},
  sj::AbstractVector{R},
) where {I <: Integer, R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3, V4, VI <: AbstractArray{<:Integer}} =
  ShiftedIndBallL0Box(ψ.h, ψ.xk, sj, ψ.l, ψ.u, true, ψ.selected)

function (ψ::ShiftedIndBallL0Box)(y)
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

fun_name(ψ::ShiftedIndBallL0Box) = "shifted L0 norm ball with box indicator"
fun_expr(ψ::ShiftedIndBallL0Box) = "t ↦ χ({‖xk + sj + t‖₀ ≤ r}) + χ({sj + t .∈ [l,u]})"
fun_params(ψ::ShiftedIndBallL0Box) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "lb = $(ψ.l)\n" * " "^14 * "ub = $(ψ.u)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedIndBallL0Box{I, R, V0, V1, V2, V3, V4, VI},
  q::AbstractVector{R},
  σ::R,
) where {I <: Integer, R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3, V4, VI <: AbstractArray{<:Integer}}
  # y = ψ.xk + ψ.sj + q in-place
  copyto!(y, q)
  @inbounds for i in eachindex(y)
    y[i] += ψ.xk[i] + ψ.sj[i]
  end
  # find largest entries
  sortperm!(ψ.p, y, rev = true, by = abs) # use ψ.p as placeholder
  # set smallest to zero
  for idx in ψ.p[(ψ.h.r + 1):end]
    y[idx] = 0
  end

  # clip back to box around the base shift
  @inbounds for i in eachindex(y)
    li = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
    ui = isa(ψ.u, Real) ? ψ.u : ψ.u[i]
    v = y[i] - (ψ.xk[i] + ψ.sj[i])
    if v < li
      y[i] = li
    elseif v > ui
      y[i] = ui
    else
      y[i] = v
    end
  end

  return y
end