export ShiftedNormL2Group

mutable struct ShiftedNormL2Group{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::GroupNormL2{R}
  xk::V0
  sj::V1
  sol::V2
  shifted_twice::Bool

  function ShiftedNormL2Group(
    h::GroupNormL2{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(sj)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice)
  end
end

shifted(
  h::GroupNormL2{R},
  xk::AbstractVector{R},
) where {R <: Real} = ShiftedNormL2Group(h, xk, zero(xk), false)
shifted(
  ψ::ShiftedNormL2Group{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
ShiftedNormL2Group(ψ.h, ψ.xk, sj, true)

fun_name(ψ::ShiftedNormL2Group) = "shifted ∑ᵢ||⋅||_2 norm"
fun_expr(ψ::ShiftedNormL2Group) = "t ↦ ‖xk + sj + t‖ₚᵖ, p = 2"
fun_params(ψ::ShiftedNormL2Group) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL2Group{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  
  ψ.sol .= q + ψ.xk + ψ.sj
  for i = 1:ψ.h.g
    y[ψ.h.idx[1,i]:ψ.h.idx[2,i]] .= max(1 - σ/norm(ψ.sol[ψ.h.idx[1,i]:ψ.h.idx[2,i]]), 0) .* ψ.sol[ψ.h.idx[1,i]:ψ.h.idx[2,i]]
  end
  y .-= (ψ.xk + ψ.sj)
  return y
end
