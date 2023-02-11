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
  shifted_twice::Bool
  selected::AbstractArray{T}

  function ShiftedNormL1Box(
    h::NormL1{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    l,
    u,
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
  selected::AbstractArray{T} = 1:length(xk),
) where {R <: Real, T <: Integer} = ShiftedNormL1Box(h, xk, zero(xk), l, u, false, selected)
shifted(
  h::NormL1{R},
  xk::AbstractVector{R},
  Δ::R,
  χ::Conjugate{IndBallL1{R}},
  selected::AbstractArray{T} = 1:length(xk),
) where {R <: Real, T <: Integer} = ShiftedNormL1Box(h, xk, zero(xk), -Δ, Δ, false, selected)
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
} = ShiftedNormL1Box(ψ.h, ψ.xk, sj, ψ.l, ψ.u, true, ψ.selected)

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

# arg min yᵀDy/2 + gᵀy + λ h(x + s + y) + χ(y | [l-s, u-s])
# variable change v = x + s + y:
# arg min vᵀDv/2 + fᵀv + λ h(v) + χ(v | [l+x, u+x])
# with fᵢ = gᵢ - dᵢ(xᵢ + sᵢ)
function iprox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL1Box{R, T, V0, V1, V2, V3, V4},
  g::AbstractVector{R},
  d::AbstractVector{R},
) where {
R <: Real,
T <: Integer,
V0 <: AbstractVector{R},
V1 <: AbstractVector{R},
V2 <: AbstractVector{R},
V3,
V4,
}
  λ = ψ.λ

  for i ∈ eachindex(y)
    li = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
    ui = isa(ψ.u, Real) ? ψ.u : ψ.u[i]
    di = d[i]
    gi = g[i]
    si = ψ.sj[i]
    xi = ψ.xk[i]
    xs = xi + si

    if i ∈ ψ.selected
      left = li - si
      right = ui - si

      if abs(di) ≤ eps(R)
        # arg min gi (xi + si + yi) + λ |xi + si + yi|
        # arg min gi vi + λ |vi|
        if abs(gi) ≤ λ
          y[i] = min(max(left, -xs), right)
        else
          y[i] = (gi > 0) ? left : right
        end

      elseif di > eps(R)
        # arg min vi² + 2fi vi / di + 2λ |vi| / di + χ(vi | [li + xi, ui + xi])
        di_2 = di / 2
        lx = li + xi
        ux = ui + xi
        gi2_di = gi / di_2 # 2 gi / di
        fi2_di = gi2_di - 2 * xs # 2fi / di = 2gi / di - 2(xi + si)
        λ2_di = λ / di_2 # 2 λ / di

        val_left = lx^2 + fi2_di * lx + λ2_di * abs(lx)
        val_right = ux^2 + fi2_di * ux + λ2_di * abs(ux)
        val_min = min(val_left, val_right)
        y[i] = val_left < val_right ? left : right
        if lx ≥ zero(R)
          argmin_quad_y = -(gi + λ) / di # less expensive to solve initial problem here
          (left ≤ argmin_quad_y ≤ right) && (y[i] = argmin_quad_y)
        elseif zero(R) ≥ ux
          argmin_quad = (λ - gi) / di # less expensive to solve initial problem here
          (left ≤ argmin_quad ≤ right) && (y[i] = argmin_quad)
        else # li ≤ -xi ≤ ui, so xi + si + yi changes sign in [li - si, ui - si]
          argmin_quad_y1 = -(gi + λ) / di # candidate 1 for initial problem
          argmin_quad_y2 = (λ - gi) / di # candidate 2 for initial problem
          if left ≤ argmin_quad_y1 ≤ right
            argmin_quad_v1 = xs + argmin_quad_y1 # candidate 1 for problem with variable change
            val_min_quad1 = argmin_quad_v1^2 + fi2_di * argmin_quad_v1 + λ2_di * abs(argmin_quad_v1)
            (val_min_quad1 < val_min) && (y[i] = argmin_quad_y1)
            val_min = min(val_min_quad1, val_min)
          end
          if left ≤ argmin_quad_y2 ≤ right
            argmin_quad_v2 = xs + argmin_quad_y2 # candidate 2 for problem with variable change
            val_min_quad2 = argmin_quad_v2^2 + fi2_di * argmin_quad_v2 + λ2_di * abs(argmin_quad_v2)
            (val_min_quad2 < val_min) && (y[i] = argmin_quad_y2)
            val_min = min(val_min_quad2, val_min)
          end
          val_0 = zero(R) # if vi = 0 ,  yi = - xi - si
          (val_0 < val_min) && (y[i] = -xs)
          val_min = min(val_0, val_min)
        end

      else # di ≤ -eps(R)
        # arg max vi² + 2fi vi / di + 2λ |vi| / di + χ(vi | [li + xi, ui + xi])
        di_2 = di / 2
        gi2_di = gi / di_2 # 2 gi / di
        fi2_di = gi2_di - 2 * xs # 2fi / di = 2gi / di - 2(xi + si)
        λ2_di = λ / di_2 # 2 λ / di
        lx = li + xi
        ux = ui + xi

        val_left = lx^2 + fi2_di * lx + λ2_di * abs(lx)
        val_right = ux^2 + fi2_di * ux + λ2_di * abs(ux)
        val_max = max(val_left, val_right)
        y[i] = (val_left > val_right) ? left : right
        if li ≤ -xi ≤ ui
          val_0 = zero(R) # if vi = 0 ,  yi = - xi - si
          (val_0 > val_max) && (y[i] = -xs)
          val_max = max(val_0, val_max)
        end
      end
      
    else
      y[i] = iprox_zero(di, gi, li - si, ui - si) 
    end
  end
  return y
end