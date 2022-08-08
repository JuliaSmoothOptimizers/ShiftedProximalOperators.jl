export ShiftedNormL0Box

mutable struct ShiftedNormL0Box{
  R <: Real,
  T <: Integer,
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
  Δ::R
  shifted_twice::Bool
  selected::UnitRange{T}

  function ShiftedNormL0Box(
    h::NormL0{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    l,
    u,
    Δ::R,
    shifted_twice::Bool,
    selected::UnitRange{T}
  ) where {R <: Real}
    sol = similar(xk)
    if any(l .> u)
      error("Error: at least one lower bound is greater than the upper bound.")
    end
    new{R, T, typeof(xk), typeof(sj), typeof(sol), typeof(l), typeof(u)}(h, xk, sj, sol, l, u, Δ, shifted_twice, selected)
  end

end

shifted(h::NormL0{R}, xk::AbstractVector{R}, l, u, Δ::R, selected::UnitRange{T}) where {R <: Real, T <: Integer} =
  ShiftedNormL0Box(h, xk, zero(xk), l, u, Δ, false, selected)
shifted(
  ψ::ShiftedNormL0Box{R, T, V0, V1, V2, V3, V4},
  sj::AbstractVector{R},
) where {R <: Real, T <: Integer, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3, V4} =
  ShiftedNormL0Box(ψ.h, ψ.xk, sj, ψ.l, ψ.u, ψ.Δ, true, ψ.selected)

fun_name(ψ::ShiftedNormL0Box) = "shifted L0 pseudo-norm with box indicator"
fun_expr(ψ::ShiftedNormL0Box) = "t ↦ λ ‖xk + sj + t‖₀ + χ({sj + t .∈ [l,u]})"
fun_params(ψ::ShiftedNormL0Box) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "lb = $(ψ.l)\n" * " "^14 * "ub = $(ψ.u)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL0Box{R, T, V0, V1, V2, V3, V4},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, T <: Integer, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3, V4}
  
  c = 2 * ψ.λ * σ
  selected = ψ.selected

  for i ∈ eachindex(q)

    li = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
    ui = isa(ψ.u, Real) ? ψ.u : ψ.u[i]

    qi = q[i]
    xi = ψ.xk[i]
    si = ψ.sj[i]

    if i ∈ selected
  
      if ui - si < qi
        if li <= -xi <= ui
          if (xi + si + qi)^2 < (ui - si - qi)^2 + c
            y[i] = -(xi+si)
          else 
            y[i] = ui - si
          end
        else
          y[i] = ui - si
        end
  
      elseif li - si > qi
        if li <= -xi <= ui
          if (xi + si + qi)^2 < (li - si - qi)^2 + c
            y[i] = -(xi+si)
          else 
            y[i] = li - si
          end
        else
          y[i] = li - si
        end
  
      else 
        if li <= -xi <= ui
          if (xi + si + qi)^2 < c
            y[i] = -(xi + si)
          else 
            y[i] = qi
          end
        else
          y[i] = qi
        end
      end

    else # min ½ σ⁻¹ (y - qi)² subject to li-si ≤ y ≤ ui-si
      if li - si <= qi <= ui - si
        y[i] = qi
      else
        y[i] = abs(li-si-qi) < abs(ui-si-qi) ? li - si : ui - si
      end
    end
  end
  return y
end
