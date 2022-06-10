export ShiftedNormL0Box

mutable struct ShiftedNormL0Box{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3,
  V4
} <: ShiftedProximableFunction
  h::NormL0{R}
  xk::V0
  sj::V1
  sol::V2
  l::V3
  u::V4
  shifted_twice::Bool

  function ShiftedNormL0Box(
    h::NormL0{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    l,
    u,
    shifted_twice::Bool
  ) where {R <: Real}
    sol = similar(xk)
    if any(l .> u)
      error("Error: at least one lower bound is greater than the upper bound.")
    end
    new{R, typeof(xk), typeof(sj), typeof(sol), typeof(l), typeof(u)}(h, xk, sj, sol, l, u, shifted_twice)
  end

end

shifted(h::NormL0{R}, xk::AbstractVector{R}, l, u) where {R <: Real} =
  ShiftedNormL0Box(h, xk, zero(xk), l, u, false)

shifted(
  ψ::ShiftedNormL0Box{R, V0, V1, V2, V3, V4},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3, V4} =
  ShiftedNormL0Box(ψ.h, ψ.xk, sj, ψ.l, ψ.u, true)

fun_name(ψ::ShiftedNormL0Box) = "shifted L0 pseudo-norm with box indicator"
fun_expr(ψ::ShiftedNormL0Box) = "t ↦ λ ‖xk + sj + t‖₀ + χ({sj + t .∈ [l,u]})"
fun_params(ψ::ShiftedNormL0Box) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "lb = $(ψ.l)\n" * " "^14 * "ub = $(ψ.u)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL0Box{R, V0, V1, V2, V3, V4},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3, V4}
  
  c2 = 2 * ψ.λ * σ

  for i ∈ eachindex(q)

    li = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
    ui = isa(ψ.u, Real) ? ψ.u : ψ.u[i]

    qi = q[i]
    xi = ψ.xk[i]
    si = ψ.sj[i]
  
    if ui - si < qi
      if li <= -xi <= ui
        if (xi + si + qi)^2 < (ui - si - qi)^2 + c2 ###### AU dessus c'est ok mais en dessous faut tout changer
          y[i] = -(xi+si)
        else 
          y[i] = ui - si
        end
      else
        y[i] = ui - si
      end
  
    elseif li - si > qi
      if li <= -xi <= ui
        if (xi + si + qi)^2 < (li - si - qi)^2 + c2
          y[i] = -(xi+si)
        else 
          y[i] = li - si
        end
      else
        y[i] = li - si
      end
  
    else 
      if li <= -xi <= ui
        if (xi + si + qi)^2 < c2
          y[i] = -(xi + si)
        else 
          y[i] = qi
        end
      else
        y[i] = qi
      end
    end
  end
  return y
end
