export ShiftedCappedl1

mutable struct ShiftedCappedl1{
  R <: Real,
  S <: AbstractArray,
  T,
  Tr,
  M <: AbstractArray{T},
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::Cappedl1{R, S, T, Tr, M}
  xk::V0  # base shift (nonzero when shifting an already shifted function)
  sj::V1  # current shift
  sol::V2   # internal storage
  shifted_twice::Bool
  function ShiftedCappedl1(
    h::Cappedl1{R, S, T, Tr, M},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real, S <: AbstractArray, T, Tr, M <: AbstractArray{T}}
    sol = similar(xk)
    new{R, S, T, Tr, M, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice)
  end
end

fun_name(ψ::ShiftedCappedl1) = "shifted Cappedl1"
fun_expr(ψ::ShiftedCappedl1) = "t ↦ Cappedl1(xk + sj + t)"

shifted(
  h::Cappedl1{R, S, T, Tr, M},
  xk::AbstractVector{R},
) where {R <: Real, S <: AbstractArray, T, Tr, M <: AbstractArray{T}} =
  ShiftedCappedl1(h, xk, zero(xk), false)
shifted(
  ψ::ShiftedCappedl1{R, S, T, Tr, M, V0, V1, V2},
  sj::AbstractVector{R},
) where {
  R <: Real,
  S <: AbstractArray,
  T,
  Tr,
  M <: AbstractArray{T},
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} = ShiftedCappedl1(ψ.h, ψ.xk, sj, true)

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedCappedl1{R, S, T, Tr, M, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {
  R <: Real,
  S <: AbstractArray,
  T,
  Tr,
  M <: AbstractArray{T},
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
}
  ψ.sol .= q .+ ψ.xk .+ ψ.sj
  ψ.h.A .= ShiftedProximalOperators.reshape_array(ψ.sol, size(ψ.h.A))
  psvd_dd!(ψ.h.F, ψ.h.A, full = false)
  for i ∈ eachindex(ψ.h.F.S)
    x1 = max(ψ.h.theta, ψ.h.F.S[i])
    x2 = min(ψ.h.theta, max(0, ψ.h.F.S[i] - ψ.λ * σ))
    if (x1 - ψ.h.F.S[i])^2 / 2 + ψ.λ * σ * ψ.h.theta < (x2 - ψ.h.F.S[i])^2 / 2 + ψ.λ * σ * x2
      ψ.h.F.S[i] = x1
    else
      ψ.h.F.S[i] = x2
    end
    for j = 1:size(ψ.h.A, 1)
      ψ.h.F.U[j, i] = ψ.h.F.U[j, i] * ψ.h.F.S[i]
    end
  end
  mul!(ψ.h.A, ψ.h.F.U, ψ.h.F.Vt)
  y .= ShiftedProximalOperators.reshape_array(ψ.h.A, size(y)) .- (ψ.xk .+ ψ.sj)
  return y
end
