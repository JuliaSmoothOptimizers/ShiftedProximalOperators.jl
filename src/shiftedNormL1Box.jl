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
    if sum(l .> u) > 0 
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
  
  c = σ * ψ.λ

  for i ∈ eachindex(y)

    if isa(ψ.l, Real) && isa(ψ.u, Real)
      li = ψ.l
      ui = ψ.u
    elseif isa(ψ.l, Real)
      li = ψ.l
      ui = ψ.u[i]
    elseif isa(ψ.u, Real)
      li = ψ.l[i]
      ui = ψ.u
    else
      li = ψ.l[i]
      ui = ψ.u[i]
    end 

    opt_left = q[i] + c
    opt_right = q[i] - c
    xs = ψ.xk[i] + ψ.sj[i]

    if opt_left < -xs
      if ui < opt_left
        y[i] = ui
      elseif opt_left < li
        y[i] = li
      else
        y[i] = opt_left
      end

    elseif -xs < opt_right
      if ui < opt_right
        y[i] = ui
      elseif opt_right < li
        y[i] = li
      else
        y[i] = opt_right
      end

    else
      if ui < -xs
        y[i] = ui
      elseif -xs < li
        y[i] = li
      else
        y[i] = -xs
      end
    end

  end
  return y
end