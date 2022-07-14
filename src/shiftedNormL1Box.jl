export ShiftedNormL1Box

mutable struct ShiftedNormL1Box{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3,
  V4
} <: ShiftedProximableFunction
  h::NormL1{R}
  xk::V0
  sj::V1
  sol::V2
  l::V3
  u::V4
  shifted_twice::Bool

  function ShiftedNormL1Box(
    h::NormL1{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    l,
    u,
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(xk)
    if any(l .> u) 
      error("Error: at least one lower bound is greater than the upper bound.")
    end
    new{R, typeof(xk), typeof(sj), typeof(sol), typeof(l), typeof(u)}(h, xk, sj, sol, l, u, shifted_twice)
  end
end

shifted(h::NormL1{R}, xk::AbstractVector{R}, l, u) where {R <: Real} =
  ShiftedNormL1Box(h, xk, zero(xk), l, u, false)
shifted(
  ψ::ShiftedNormL1Box{R, V0, V1, V2, V3, V4},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3, V4} =
  ShiftedNormL1Box(ψ.h, ψ.xk, sj, ψ.l, ψ.u, true)

fun_name(ψ::ShiftedNormL1Box) = "shifted L1 norm with box indicator"
fun_expr(ψ::ShiftedNormL1Box) = "t ↦ ‖xk + sj + t‖₁ + χ({sj + t .∈ [l,u]})"
fun_params(ψ::ShiftedNormL1Box) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "l = $(ψ.l)\n" * " "^14 * "u = $(ψ.u)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL1Box{R, V0, V1, V2, V3, V4},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3, V4}
  
  c = 2 * σ * ψ.λ

  for i ∈ eachindex(y)

    li = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
    ui = isa(ψ.u, Real) ? ψ.u : ψ.u[i] 

    opt_left = q[i] + c/2
    opt_right = q[i] - c/2
    xi = ψ.xk[i]
    si = ψ.sj[i]

    if opt_left < -(xi + si)
      if ui - si < opt_left
        y[i] = ui - si
      elseif opt_left < li - si
        y[i] = li - si
      else
        y[i] = opt_left
      end

    elseif -(xi + si) < opt_right
      if ui - si < opt_right
        y[i] = ui - si
      elseif opt_right < li - si
        y[i] = li - si
      else
        y[i] = opt_right
      end

    else
      if ui - si < -(xi + si)
        y[i] = ui - si
      elseif -(xi + si) < li - si
        y[i] = li - si
      else
        y[i] = -(xi + si)
      end
    end

  end
  return y
end