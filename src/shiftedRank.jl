export ShiftedRank

mutable struct ShiftedRank{
    R <: Real,
    V0 <: AbstractVector{R},
    V1 <: AbstractVector{R},
    V2 <: AbstractVector{R},
  } <: ShiftedProximableFunction
    h::Rank{R}
    xk::V0  # base shift (nonzero when shifting an already shifted function)
    sj::V1  # current shift
    sol::V2   # internal storage
    shifted_twice::Bool

    function ShiftedRank(
      h::Rank{R},
      xk::AbstractVector{R},
      sj::AbstractVector{R},
      shifted_twice::Bool,
      ) where {R <: Real}
      sol = similar(xk)
      new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice)
    end
end

shifted(h::Rank{R}, xk::AbstractVector{R}) where {R <: Real} =
ShiftedRank(h, xk, zero(xk), false)
shifted(
    ψ::ShiftedRank{R, V0, V1, V2},
    sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
ShiftedRank(ψ.h, ψ.xk, sj, true)

fun_name(ψ::ShiftedRank) = "shifted Rank" 
fun_expr(ψ::ShiftedRank) = "t ↦ rank(xk + sj + t)"
fun_params(ψ::ShiftedRank) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)"

function prox!(
    y::AbstractVector{R},
    ψ::ShiftedRank{R, V0, V1, V2},
    q::AbstractVector{R},
    σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
    Q = reshape(q + ψ.xk + ψ.sj, ψ.h.nrow, ψ.h.ncol)
    SQ = svd(Q)
    yvec = SQ.S
    c = sqrt(2 * ψ.λ * σ)

    for i ∈ eachindex(SQ.S)
        if abs(yvec[i]) <= c
            yvec[i] = 0
        end
    end
    
    return vec(reshape(SQ.U * Diagonal(yvec) * SQ.Vt - reshape(ψ.xk + ψ.sj, ψ.h.nrow, ψ.h.ncol), ψ.h.nrow * ψ.h.ncol, 1))
end