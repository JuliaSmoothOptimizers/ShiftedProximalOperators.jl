export ShiftedNormL1Box

mutable struct ShiftedNormL1Box{
  R <: Real,
  T <: Integer,
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
  Δ::R
  shifted_twice::Bool
  selected::UnitRange{T}

  function ShiftedNormL1Box(
    h::NormL1{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    l,
    u,
    Δ::R,
    shifted_twice::Bool,
    selected::UnitRange{T},
  ) where {R <: Real, T <: Integer}
    sol = similar(xk)
    if any(l .> u) 
      error("Error: at least one lower bound is greater than the upper bound.")
    end
    new{R, T, typeof(xk), typeof(sj), typeof(sol), typeof(l), typeof(u)}(h, xk, sj, sol, l, u, Δ, shifted_twice, selected)
  end
end

shifted(h::NormL1{R}, xk::AbstractVector{R}, l, u, Δ::R) where {R <: Real} =
  ShiftedNormL1Box(h, xk, zero(xk), l, u, Δ, false, 1:length(xk))
shifted(h::NormL1{R}, xk::AbstractVector{R}, l, u, Δ::R, selected::UnitRange{T}) where {R <: Real, T <: Integer} =
  ShiftedNormL1Box(h, xk, zero(xk), l, u, Δ, false, selected)
shifted(
  ψ::ShiftedNormL1Box{R, T, V0, V1, V2, V3, V4},
  sj::AbstractVector{R},
) where {R <: Real, T <: Integer, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3, V4} =
  ShiftedNormL1Box(ψ.h, ψ.xk, sj, ψ.l, ψ.u, ψ.Δ, true, ψ.selected)

(ψ::ShiftedNormL1Box)(y) = ψ.h((ψ.xk + ψ.sj + y)[ψ.selected]) + (all(ψ.l .<= ψ.sj + y .<= ψ.u .|| ψ.l .≈ ψ.sj + y .|| ψ.u .≈ ψ.sj + y) ? 0 : Inf)

fun_name(ψ::ShiftedNormL1Box) = "shifted L1 norm with box indicator"
fun_expr(ψ::ShiftedNormL1Box) = "t ↦ ‖xk + sj + t‖₁ + χ({sj + t .∈ [l,u]})"
fun_params(ψ::ShiftedNormL1Box) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "l = $(ψ.l)\n" * " "^14 * "u = $(ψ.u)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL1Box{R, T, V0, V1, V2, V3, V4},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, T <: Integer, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3, V4}
  
  c = 2 * σ * ψ.λ
  selected = ψ.selected

  for i ∈ eachindex(y)

    li = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
    ui = isa(ψ.u, Real) ? ψ.u : ψ.u[i] 

    qi = q[i]
    opt_left = qi + c/2
    opt_right = qi - c/2
    xi = ψ.xk[i]
    si = ψ.sj[i]

    if i ∈ selected

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