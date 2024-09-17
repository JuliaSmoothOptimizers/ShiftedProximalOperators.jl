export ShiftedNormL1

mutable struct ShiftedNormL1{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::NormL1{R}
  xk::V0
  sj::V1
  sol::V2
  shifted_twice::Bool
  xsy::V2

  function ShiftedNormL1(
    h::NormL1{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(xk)
    xsy = similar(xk)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice, xsy)
  end
end

shifted(h::NormL1{R}, xk::AbstractVector{R}) where {R <: Real} =
  ShiftedNormL1(h, xk, zero(xk), false)
shifted(
  ψ::ShiftedNormL1{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormL1(ψ.h, ψ.xk, sj, true)

fun_name(ψ::ShiftedNormL1) = "shifted L1 norm"
fun_expr(ψ::ShiftedNormL1) = "t ↦ ‖xk + sk + t‖₁"
fun_params(ψ::ShiftedNormL1) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL1{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  λ = ψ.h.lambda
  @. y = -ψ.xk - ψ.sj

  for i ∈ eachindex(y)
    y[i] = min(max(y[i], q[i] - λ * σ), q[i] + λ * σ)
  end

  return y
end

# arg min yᵀDy/2 - gᵀy + λ h(x + s + y)
# variable change v = x + s + y:
# arg min vᵀDv/2 - fᵀv + λ h(v)
# with fᵢ = gᵢ + dᵢ(xᵢ + sᵢ)
function iprox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL1{R, V0, V1, V2},
  g::AbstractVector{R},
  d::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  λ = ψ.h.lambda
  @. y = -ψ.xk - ψ.sj

  for i ∈ eachindex(y)
    if d[i] < 0
      y[i] = - 1/eps(R) 
      continue
    else 
      y[i] = min(max(y[i], -g[i] / d[i] - λ / d[i]), -g[i] / d[i] + λ / d[i])
    end
  end

  return y
end
