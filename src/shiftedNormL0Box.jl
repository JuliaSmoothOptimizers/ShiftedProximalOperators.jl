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

function solve_ith_subproblem_proxL0(
  li::R,
  ui::R,
  xi::R,
  si::R,
  qi::R,
  xs::R, # xi + si
  sq::R, # xi + qi
  xsq::R, # xi + si + qi
  ci::R,
) where {R <: Real}
  # yi = arg min (yi - qi)^2 + ci * ||xi + si + yi||₀ + χ(si + yi | [li, ui])
  # possible minima locations:
  # yi = li - si
  # yi = ui - si
  # yi = -xi - si, if: li + xi ≤ 0 ≤ ui + xi, leads to h(xi + si + yi) = 0
  # yi = qi, if: di > 0 and li + xi ≤ xi + si + qi ≤ ui + xi
  val_left = (li - sq)^2 + (xi == -li ? 0 : ci) # left: yi = li - si
  val_right = (ui - sq)^2 + (xi == -ui ? 0 : ci) # right: yi = ui - si
  yi = val_left < val_right ? (li - si) : (ui - si)
  val_min = min(val_left, val_right)
  if li ≤ -xi ≤ ui  # <=> li + xi ≤ 0 ≤ ui + xi
    # compute (xi + si + qi)^2 with y = -xi - si so that h(xi + si + y) = 0
    val_0 = xsq^2
    val_0 < val_min && (yi = -xs)
    val_min = min(val_0, val_min)
  end
  if li ≤ sq ≤ ui  # <=> li + xi ≤ xi + si + qi ≤ ui + xi
    # if yi = qi then the val is λ,
    # except if xi + si + qi = 0 because in this case h(xi + si + qi) = 0
    val_xsq = xsq == 0 ? zero(R) : ci
    val_xsq < val_min && (yi = qi)
  end
  return yi
end

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
      # yi = arg min (yi - qi)^2 + 2λ||xi + si + yi||₀ / di + χ(si + yi | [li, ui])
      y[i] = solve_ith_subproblem_proxL0(li, ui, xi, si, qi, xs, sq, xsq, c)
    else # min ½ σ⁻¹ (y - qi)² subject to li - si ≤ y ≤ ui - si
      y[i] = prox_zero(qi, li - si, ui - si)
    end
  end
  return y
end

function solve_ith_subproblem_iproxL0_neg(
  li::R,
  ui::R,
  xi::R,
  si::R,
  xs::R, # xi + si
  sq::R, # xi + qi
  xsq::R, # xi + si + qi
  ci::R,
) where {R <: Real}
  # yi = arg max (yi - qi)^2 + ci||xi + si + yi||₀ - χ(si + yi | [li, ui])
  # where ci < 0 (ci = 2λ / di)
  # possible maxima locations:
  # yi = li - si
  # yi = ui - si
  # yi = -xi - si, if: li + xi ≤ 0 ≤ ui + xi, leads to h(xi + si + yi) = 0
  val_left = (li - sq)^2 + (xi == -li ? 0 : ci) # left: yi = li - si
  val_right = (ui - sq)^2 + (xi == -ui ? 0 : ci) # right: yi = ui - si
  yi = val_left > val_right ? (li - si) : (ui - si)
  val_max = max(val_left, val_right)
  if li ≤ -xi ≤ ui  # <=> li + xi ≤ 0 ≤ ui + xi
    # compute (xi + si + qi)^2 with y = -xi - si so that h(xi + si + y) = 0
    val_0 = xsq^2
    val_0 > val_max && (yi = -xs)
  end
  return yi
end

function iprox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL0Box{R, T, V0, V1, V2, V3, V4},
  q::AbstractVector{R},
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
  λ2 = 2 * ψ.λ

  for i ∈ eachindex(q)
    li = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
    ui = isa(ψ.u, Real) ? ψ.u : ψ.u[i]
    di = d[i]
    ci = λ2 / di

    qi = q[i]
    si = ψ.sj[i]
    sq = si + qi

    # yi = arg min di * (yi - qi)^2 /2 + h(xi + si + yi) + χ(si + yi | [li, ui])
    if i ∈ ψ.selected
      xi = ψ.xk[i]
      xs = xi + si
      xsq = xs + qi
      if di > eps(R)
        # yi = arg min (yi - qi)^2 + 2λ||xi + si + yi||₀ / di + χ(si + yi | [li, ui])
        y[i] = solve_ith_subproblem_proxL0(li, ui, xi, si, qi, xs, sq, xsq, ci)
      elseif di < -eps(R)
        # yi = arg max (yi - qi)^2 + 2λ||xi + si + yi||₀ / di - χ(si + yi | [li, ui])
        y[i] = solve_ith_subproblem_iproxL0_neg(li, ui, xi, si, xs, sq, xsq, ci) 
      else # abs(di) < eps(R) (consider di = 0 in this case)
        # yi = arg min h(xi + si + yi) + χ(si + yi | [li, ui])
        y[i] = (li ≤ -xi ≤ ui) ? -xs : zero(R)
        # maybe set something else than 0
      end
    else # min ½ di⁻¹ (y - qi)² subject to li - si ≤ y ≤ ui - si
      y[i] = iprox_zero(qi, li, ui, si, di)
    end
  end
  return y
end

function iprox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL0Box{R, T, V0, V1, V2, V3, V4},
  q::AbstractVector{R},
  d::R,
) where {
  R <: Real,
  T <: Integer,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3,
  V4,
}

  if d > eps(R)
    prox!(y, ψ, q, d)
  elseif d < - eps(R)
    c = 2 * ψ.λ / d
    for i ∈ eachindex(q)
      li = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
      ui = isa(ψ.u, Real) ? ψ.u : ψ.u[i]
      qi = q[i]
      si = ψ.sj[i]
      sq = si + qi
      # yi = arg min d * (yi - qi)^2 /2 + h(xi + si + yi) + χ(si + yi | [li, ui])
      if i ∈ ψ.selected
        xi = ψ.xk[i]
        xs = xi + si
        xsq = xs + qi
        # yi = arg max (yi - qi)^2 + 2λ||xi + si + yi||₀ / d - χ(si + yi | [li, ui])
        y[i] = solve_ith_subproblem_iproxL0_neg(li, ui, xi, si, xs, sq, xsq, c) 
      else # min ½ d⁻¹ (y - qi)² subject to li - si ≤ y ≤ ui - si
        y[i] = negative_prox_zero(qi, li - si, ui - si)
      end
    end
  else # abs(di) < eps(R) (consider di = 0 in this case)
    for i ∈ eachindex(q)
      li = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
      ui = isa(ψ.u, Real) ? ψ.u : ψ.u[i]
      si = ψ.sj[i]
      # yi = arg min di * (yi - qi)^2 /2 + h(xi + si + yi) + χ(si + yi | [li, ui])
      if i ∈ ψ.selected
        xi = ψ.xk[i]
        xs = xi + si
        # yi = arg min h(xi + si + yi) + χ(si + yi | [li, ui])
        y[i] = (li ≤ -xi ≤ ui) ? -xs : zero(R)
        # maybe set something else than 0
      else # min 0 subject to li - si ≤ y ≤ ui - si
        y[i] = zero(R)
      end
    end
  end

  return y
end
