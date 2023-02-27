export ShiftedRootNormLhalfBox

mutable struct ShiftedRootNormLhalfBox{
  R <: Real,
  T <: Integer,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3,
  V4,
} <: ShiftedProximableFunction
  h::RootNormLhalf{R}
  xk::V0
  sj::V1
  sol::V2
  l::V3
  u::V4
  shifted_twice::Bool
  selected::AbstractArray{T}

  function ShiftedRootNormLhalfBox(
    h::RootNormLhalf{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    l,
    u,
    shifted_twice::Bool,
    selected::AbstractArray{T},
  ) where {R <: Real, T <: Integer}
    sol = similar(sj)
    new{R, T, typeof(xk), typeof(sj), typeof(sol), typeof(l), typeof(u)}(
      h,
      xk,
      sj,
      sol,
      l,
      u,
      shifted_twice,
      selected,
    )
  end
end

shifted(
  h::RootNormLhalf{R},
  xk::AbstractVector{R},
  l,
  u,
  selected::AbstractArray{T} = 1:length(xk),
) where {R <: Real, T <: Integer} = ShiftedRootNormLhalfBox(h, xk, zero(xk), l, u, false, selected)
shifted(
  h::RootNormLhalf{R},
  xk::AbstractVector{R},
  Δ::R,
  χ::Conjugate{IndBallL1{R}},
  selected::AbstractArray{T} = 1:length(xk),
) where {R <: Real, T <: Integer} = ShiftedRootNormLhalfBox(h, xk, zero(xk), -Δ, Δ, false, selected)
shifted(
  ψ::ShiftedRootNormLhalfBox{R, T, V0, V1, V2, V3, V4},
  sj::AbstractVector{R},
) where {
  R <: Real,
  T <: Integer,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3,
  V4,
} = ShiftedRootNormLhalfBox(ψ.h, ψ.xk, sj, ψ.l, ψ.u, true, ψ.selected)

function (ψ::ShiftedRootNormLhalfBox)(y)
  val = ψ.h((ψ.xk + ψ.sj + y)[ψ.selected]) # use views here?
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

fun_name(ψ::ShiftedRootNormLhalfBox) = "shifted ∑ᵢ√|⋅| norm with L∞-norm box indicator"
fun_expr(ψ::ShiftedRootNormLhalfBox) = "t ↦ ‖xk + sj + t‖ₚᵖ + χ({sj + t .∈ [l,u]}), p = 1/2"
fun_params(ψ::ShiftedRootNormLhalfBox) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "lb = $(ψ.l)\n" * " "^14 * "ub = $(ψ.u)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedRootNormLhalfBox{R, T, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {
  R <: Real,
  T <: Integer,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
}
  ϕ(z) = acos(σ * ψ.λ / 4 * (abs(z) / 3)^(-3 / 2) + 0im)

  ψ.sol .= (ψ.xk .+ ψ.sj)
  RNorm(tt, l) = (tt - q[l])^2 / 2 / σ + ψ.λ * sqrt(abs(tt + ψ.sol[l]))
  for i ∈ eachindex(q)
    li = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
    ui = isa(ψ.u, Real) ? ψ.u : ψ.u[i]
    xi = ψ.xk[i]
    si = ψ.sj[i]
    qi = q[i]

    if i ∈ ψ.selected
      xs = ψ.sol[i]
      xsq = xs + qi
      val = real(2 * sign(xsq) / 3 * abs(xsq) * (1 + cos(2 * π / 3 - 2 * ϕ(xsq) / 3)))

      (_, a) = findmin((
        RNorm(li - si, i),
        RNorm(ui - si, i),
        (li ≤ -xi ≤ ui) ? RNorm(-xs, i) : Inf,
        (li ≤ val - xi ≤ ui) ? RNorm(val - xs, i) : Inf,
      ))
      y[i] = a == 1 ? (li - si) : a == 2 ? (ui - si) : a == 3 ? -xs : (val - xs)
    else
      y[i] = prox_zero(qi, li - si, ui - si)
    end
  end
  return y
end
