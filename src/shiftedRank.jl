export ShiftedRank

mutable struct ShiftedRank{
  R <: Real,
  S <: AbstractArray,
  T,
  Tr,
  M <: AbstractArray{T},
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::Rank{R, S, T, Tr, M}
  xk::V0  # base shift (nonzero when shifting an already shifted function)
  sj::V1  # current shift
  sol::V2   # internal storage
  shifted_twice::Bool
  function ShiftedRank(
    h::Rank{R, S, T, Tr, M},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real, S <: AbstractArray, T, Tr, M <: AbstractArray{T}}
    sol = similar(xk)
    new{R, S, T, Tr, M, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice)
  end
end

fun_name(ψ::ShiftedRank) = "shifted Rank"
fun_expr(ψ::ShiftedRank) = "t ↦ rank(xk + sj + t)"

shifted(
  h::Rank{R, S, T, Tr, M},
  xk::AbstractVector{R},
) where {R <: Real, S <: AbstractArray, T, Tr, M <: AbstractArray{T}} =
  ShiftedRank(h, xk, zero(xk), false)
shifted(
  ψ::ShiftedRank{R, S, T, Tr, M, V0, V1, V2},
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
} = ShiftedRank(ψ.h, ψ.xk, sj, true)

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedRank{R, S, T, Tr, M, V0, V1, V2},
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
  ψ.h.A .= reshape_array(ψ.sol, size(ψ.h.A))
  psvd_dd!(ψ.h.F, ψ.h.A, full = false)
  c = sqrt(2 * ψ.λ * σ)
  for i ∈ eachindex(ψ.h.F.S)
    if ψ.h.F.S[i] <= c
      ψ.h.F.U[:, i] .= 0
    else
      for j = 1:size(ψ.h.A, 1)
        ψ.h.F.U[j, i] = ψ.h.F.U[j, i] .* ψ.h.F.S[i]
      end
    end
  end
  mul!(ψ.h.A, ψ.h.F.U, ψ.h.F.Vt)
  y .= reshape_array(ψ.h.A, size(y)) .- (ψ.xk .+ ψ.sj)
  return y
end
