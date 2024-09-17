export ShiftedNormL0

mutable struct ShiftedNormL0{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::NormL0{R}
  xk::V0  # base shift (nonzero when shifting an already shifted function)
  sj::V1  # current shift
  sol::V2   # internal storage
  shifted_twice::Bool
  xsy::V2
  function ShiftedNormL0(
    h::NormL0{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(xk)
    xsy = similar(xk)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice, xsy)
  end
end

fun_name(ψ::ShiftedNormL0) = "shifted L0 pseudo-norm"
fun_expr(ψ::ShiftedNormL0) = "t ↦ ‖xk + sj + t‖₀"

shifted(h::NormL0{R}, xk::AbstractVector{R}) where {R <: Real} =
  ShiftedNormL0(h, xk, zero(xk), false)
shifted(
  ψ::ShiftedNormL0{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormL0(ψ.h, ψ.xk, sj, true)

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL0{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  λ = ψ.h.lambda
  c = sqrt(2 * λ * σ)
  for i ∈ eachindex(q)
    xps = ψ.xk[i] + ψ.sj[i]
    if abs(xps + q[i]) ≤ c
      y[i] = -xps
    else
      y[i] = q[i]
    end
  end
  return y
end

# arg min yᵀDy/2 - gᵀy + λ h(x + s + y)
# variable change v = x + s + y:
# arg min vᵀDv/2 - fᵀv + λ h(v)
# with fᵢ = gᵢ + dᵢ(xᵢ + sᵢ)
function iprox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL0{R, V0, V1, V2},
  g::AbstractVector{R},
  d::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  λ = ψ.h.lambda
  for i ∈ eachindex(y)
    di = d[i]
    if d[i] ≤ 0
      y[i] = - 1/eps(R) 
      continue
    else 
      ci = sqrt(2 * λ * di)
      xps = ψ.xk[i] + ψ.sj[i]
      if abs(di * xps - g[i]) ≤ ci
        y[i] = -xps
      else
        y[i] = -g[i] / di
      end
    end
  end
  return y
end
