export ShiftedNormL0Box

mutable struct ShiftedNormL0Box{
  R <: Real,
  T <: Integer,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3,
  V4,
} <: ShiftedProximableFunction
  h::NormL0{R}
  xk::V0
  sj::V1
  sol::V2
  l::V3
  u::V4
  shifted_twice::Bool
  selected::AbstractArray{T}

  function ShiftedNormL0Box(
    h::NormL0{R},
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
  h::NormL0{R},
  xk::AbstractVector{R},
  l,
  u,
  selected::AbstractArray{T} = 1:length(xk),
) where {R <: Real, T <: Integer} = ShiftedNormL0Box(h, xk, zero(xk), l, u, false, selected)
shifted(
  h::NormL0{R},
  xk::AbstractVector{R},
  Δ::R,
  χ::Conjugate{IndBallL1{R}},
  selected::AbstractArray{T} = 1:length(xk),
) where {R <: Real, T <: Integer} = ShiftedNormL0Box(h, xk, zero(xk), -Δ, Δ, false, selected)
shifted(
  ψ::ShiftedNormL0Box{R, T, V0, V1, V2, V3, V4},
  sj::AbstractVector{R},
) where {
  R <: Real,
  T <: Integer,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3,
  V4,
} = ShiftedNormL0Box(ψ.h, ψ.xk, sj, ψ.l, ψ.u, true, ψ.selected)

function (ψ::ShiftedNormL0Box)(y)
  val = ψ.h((ψ.xk + ψ.sj + y)[ψ.selected])
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

fun_name(ψ::ShiftedNormL0Box) = "shifted L0 pseudo-norm with box indicator"
fun_expr(ψ::ShiftedNormL0Box) = "t ↦ λ ‖xk + sj + t‖₀ + χ({sj + t .∈ [l,u]})"
fun_params(ψ::ShiftedNormL0Box) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "lb = $(ψ.l)\n" * " "^14 * "ub = $(ψ.u)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL0Box{R, T, V0, V1, V2, V3, V4},
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
  c = 2 * ψ.λ * σ

  for i ∈ eachindex(q)
    li = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
    ui = isa(ψ.u, Real) ? ψ.u : ψ.u[i]

    qi = q[i]
    si = ψ.sj[i]
    sq = si + qi

    if i ∈ ψ.selected
      xi = ψ.xk[i]
      xs = xi + si
      xsq = xs + qi
      val_left = (li - sq)^2 + (xi == -li ? 0 : c)
      val_right = (ui - sq)^2 + (xi == -ui ? 0 : c)
      # subtract x + s from solution explicitly here instead of doing it
      # numerically at the end
      y[i] = val_left < val_right ? (li - si) : (ui - si)
      val_min = min(val_left, val_right)
      if li ≤ -xi ≤ ui  # <=> li + xi ≤ 0 ≤ ui + xi
        val_0 = xsq^2
        val_0 < val_min && (y[i] = -xs)
        val_min = min(val_0, val_min)
      end
      if li ≤ sq ≤ ui  # <=> li + xi ≤ xsq ≤ ui + xi
        val_xsq = xsq == 0 ? zero(R) : c
        val_xsq < val_min && (y[i] = qi)
      end

    else # min ½ σ⁻¹ (y - qi)² subject to li - si ≤ y ≤ ui - si
      y[i] = prox_zero(qi, li - si, ui - si)
    end
  end
  return y
end

# arg min yᵀDy/2 + gᵀy + λ h(x + s + y) + χ(y | [l-s, u-s]) 
function iprox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL0Box{R, T, V0, V1, V2, V3, V4},
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

      if abs(di) < eps(R) # consider di == 0
        if gi == zero(R)
          y[i] = (li ≤ -xi ≤ ui) ? -xs : zero(R)
        else
          if gi > zero(R)
            left = li - si
            val_min = gi * left + (xi == -li ? 0 : λ)
            y[i] = left
          elseif gi < zero(R)
            right = ui - si
            val_min = gi * right + (xi == -ui ? 0 : λ)
            y[i] = right
          end
          # check value when h(xi+si+yi) = 0
          if li ≤ -xi ≤ ui  # <=> li + xi ≤ 0 ≤ ui + xi
            val_0 = - gi * xs
            (val_0 < val_min) && (y[i] = -xs)
            val_min = min(val_0, val_min)
          end        
        end

      else # di != 0
        di_2 = di / 2
        left = li - si
        right = ui - si
        gi2_di = gi / di_2 # 2 gi / di
        λ2_di = λ / di_2 # 2 λ / di

        if di ≥ eps(R)
          # arg min yi² + 2giyi / di + 2λ||xi + si + yi||₀ / di + χ(si + yi | [li, ui])
          argmin_quad = -gi / di
          if left ≤ argmin_quad ≤ right  # <=> li - si ≤ -gi / di ≤ ui - si
            if gi == 0
              val_min = xs == 0 ? zero(R) : λ2_di
            else
              # (gi / di)² - 2 gi * (gi / di) / di + 2λ||xi + si + yi||₀ / di
              # = - (gi / di)^2 + 2λ||xi + si + yi||₀ / di
              val_min = (argmin_quad + xs == 0) ? (-argmin_quad^2) : (-argmin_quad^2 + λ2_di)
            end
            y[i] = argmin_quad
          else
            val_left = left^2 + gi2_di * left + (xi == -li ? 0 : λ2_di)
            val_right = right^2 + gi2_di * right + (xi == -ui ? 0 : λ2_di)
            y[i] = (val_left < val_right) ? left : right
            val_min = min(val_left, val_right)
          end
          # check value when h(xi+si+yi) = 0
          if li ≤ -xi ≤ ui  # <=> li + xi ≤ 0 ≤ ui + xi 
            val_0 = xs^2 - gi2_di * xs # (xi + si)^2 - 2gi (xi + si) / di
            (val_0 < val_min) && (y[i] = -xs)
            val_min = min(val_0, val_min)
          end

        else # di ≤ eps(R)
          # arg max yi² + 2giyi / di + 2λ||xi + si + yi||₀ / di - χ(si + yi | [li, ui])
          val_left = left^2 + gi2_di * left + (xi == -li ? 0 : λ2_di)
          val_right = right^2 + gi2_di * right + (xi == -ui ? 0 : λ2_di)
          y[i] = (val_left > val_right) ? left : right
          val_max = max(val_left, val_right)
          # check value when h(xi+si+yi) = 0
          if li ≤ -xi ≤ ui  # <=> li + xi ≤ 0 ≤ ui + xi 
            val_0 = xs^2 - gi2_di * xs # (xi + si)^2 - 2gi (xi + si) / di
            (val_0 > val_max) && (y[i] = -xs)
            val_max = max(val_0, val_max)
          end
        end
      end

    else
      y[i] = iprox_zero(di, gi, li - si, ui - si) 
    end
  end
  return y
end