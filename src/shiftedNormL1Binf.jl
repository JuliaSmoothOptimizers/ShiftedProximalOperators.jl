export ShiftedNormL1BInf

mutable struct ShiftedNormL1BInf{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3 <: Union{R, AbstractVector{R}},
  V4 <: Union{R, AbstractVector{R}}
} <: ShiftedProximableFunction
  h::NormL1{R}
  xk::V0
  sj::V1
  sol::V2
  l::V3
  u::V4
  shifted_twice::Bool

  function ShiftedNormL1BInf(
    h::NormL1{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    l::Union{R, AbstractVector{R}},
    u::Union{R, AbstractVector{R}},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(xk)
    if sum(l .> u) > 0 
      error("Error on the trust region bounds, at least one lower bound is greater than the upper bound.")
    else
      new{R, typeof(xk), typeof(sj), typeof(sol), typeof(l), typeof(u)}(h, xk, sj, sol, l, u, shifted_twice)
    end
  end
end

# We cannot use this function anymore with [l,u] trust region 
#=
(ψ::ShiftedNormL1BInf)(y) = ψ.h(ψ.xk + ψ.sj + y) + IndBallLinf(ψ.Δ)(ψ.sj + y)
=#

shifted(h::NormL1{R}, xk::AbstractVector{R}, l::Union{R, AbstractVector{R}}, u::Union{R, AbstractVector{R}}) where {R <: Real} =
  ShiftedNormL1BInf(h, xk, zero(xk), l, u, false)
shifted(
  ψ::ShiftedNormL1BInf{R, V0, V1, V2, V3, V4},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3 <: Union{R, AbstractVector{R}}, V4 <: Union{R, AbstractVector{R}}} =
  ShiftedNormL1BInf(ψ.h, ψ.xk, sj, ψ.l, ψ.u, true)

fun_name(ψ::ShiftedNormL1BInf) = "shifted L1 norm with generalized trust region indicator"
fun_expr(ψ::ShiftedNormL1BInf) = "t ↦ ‖xk + sj + t‖₁ + χ({sj + t .∈ [l,u]})"
fun_params(ψ::ShiftedNormL1BInf) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "l = $(ψ.l)\n" * " "^14 * "u = $(ψ.u)"


function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL1BInf{R, V0, V1, V2, V3, V4},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3 <: Union{R, AbstractVector{R}}, V4 <: Union{R, AbstractVector{R}}}
  
  c = σ * ψ.λ

  for i ∈ eachindex(y)

    if isa(ψ.l, Real) & isa(ψ.u, Real)
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