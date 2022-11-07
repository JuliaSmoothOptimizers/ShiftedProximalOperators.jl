export ShiftedNormL1Box

mutable struct ShiftedNormL1Box{
  R <: Real,
  T <: Integer,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3,
  V4,
} <: ShiftedProximableFunction
  h::NormL1{R}
  xk::V0
  sj::V1
  sol::V2
  l::V3
  u::V4
  Δ::R
  shifted_twice::Bool
  selected::AbstractArray{T}

  function ShiftedNormL1Box(
    h::NormL1{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    l,
    u,
    Δ::R,
    shifted_twice::Bool,
    selected::AbstractArray{T},
  ) where {R <: Real, T <: Integer}
    sol = similar(xk)
    if any(l .> u)
      error("Error: at least one lower bound is greater than the upper bound.")
    end
    new{R, T, typeof(xk), typeof(sj), typeof(sol), typeof(l), typeof(u)}(
      h,
      xk,
      sj,
      sol,
      l,
      u,
      Δ,
      shifted_twice,
      selected,
    )
  end
end

shifted(
  h::NormL1{R},
  xk::AbstractVector{R},
  l,
  u,
  Δ::R,
  selected::AbstractArray{T} = 1:length(xk),
) where {R <: Real, T <: Integer} = ShiftedNormL1Box(h, xk, zero(xk), l, u, Δ, false, selected)
shifted(
  ψ::ShiftedNormL1Box{R, T, V0, V1, V2, V3, V4},
  sj::AbstractVector{R},
) where {
  R <: Real,
  T <: Integer,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3,
  V4,
} = ShiftedNormL1Box(ψ.h, ψ.xk, sj, ψ.l, ψ.u, ψ.Δ, true, ψ.selected)

function (ψ::ShiftedNormL1Box)(y)
  val = ψ.h((ψ.xk + ψ.sj + y)[ψ.selected])
  ϵ = √eps(eltype(y))
  for i ∈ eachindex(y)
    lower = typeof(ψ.l) <: Real ? ψ.l : ψ.l[i]
    upper = typeof(ψ.u) <: Real ? ψ.u : ψ.u[i]
    if !(lower - ϵ ≤ ψ.sj[i] + y[i] ≤ upper + ϵ)
      return Inf
    end
  end
  return val
end

fun_name(ψ::ShiftedNormL1Box) = "shifted L1 norm with box indicator"
fun_expr(ψ::ShiftedNormL1Box) = "t ↦ ‖xk + sj + t‖₁ + χ({sj + t .∈ [l,u]})"
fun_params(ψ::ShiftedNormL1Box) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "l = $(ψ.l)\n" * " "^14 * "u = $(ψ.u)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL1Box{R, T, V0, V1, V2, V3, V4},
  q::AbstractVector{R},
  σ::R,
) where {
  R <: Real,
  T <: Integer,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3,
  V4,
}
  σλ = σ * ψ.λ

  for i ∈ eachindex(y)
    li = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
    ui = isa(ψ.u, Real) ? ψ.u : ψ.u[i]

    qi = q[i]
    si = ψ.sj[i]
    sq = si + qi

    if i ∈ ψ.selected
      xi = ψ.xk[i]
      xs = xi + si
      xsq = xs + qi

      y[i] = if xsq ≤ -σλ
        qi + σλ
      elseif xsq ≥ σλ
        qi - σλ
      else
        -xs
      end
      y[i] = min(max(y[i], li - si), ui - si)

    else # min ½ σ⁻¹ (y - qi)² subject to li-si ≤ y ≤ ui-si
      y[i] = prox_zero(qi, li - si, ui - si)
    end
  end
  return y
end
