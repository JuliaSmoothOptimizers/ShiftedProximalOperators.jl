export ShiftedNormLinf

"""
Allows to compute the ‖.‖∞ operator with variable bounds: t ↦ λ ‖xk + sj + t‖∞

The proximal operator is also provided. To do so, we use the algorithm proposed in Efficient Projections onto the ℓ1-Ball for Learning in High Dimensions¹

¹https://ai.stanford.edu/~jduchi/projects/jd_ss_ys_l1.pdf
"""

mutable struct ShiftedNormLinf{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::Conjugate{IndBallL1{R}}
  xk::V0
  sj::V1
  sol::V2
  shifted_twice::Bool
  xsy::V2

  function ShiftedNormLinf(
    h::Conjugate{IndBallL1{R}},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(sj)
    xsy = similar(sj)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice, xsy)
  end
end

shifted(h::Conjugate{IndBallL1{R}}, xk::AbstractVector{R}) where {R <: Real} = 
    ShiftedNormLinf(h, xk, zero(xk), false)
shifted(
  ψ::ShiftedNormLinf{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormLinf(ψ.h, ψ.xk, sj, true)

fun_name(ψ::ShiftedNormLinf) = "shifted L∞ norm"
fun_expr(ψ::ShiftedNormLinf) = "t ↦ λ ‖xk + sj + t‖∞"
fun_params(ψ::ShiftedNormLinf) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14


function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormLinf{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  λ = ψ.h.f.r
  @. ψ.sol = q + ψ.xk + ψ.sj

  r = σ * λ
  y .= ψ.sol .- _proj_l1ball(ψ.sol, r)
  @. y -= (ψ.xk + ψ.sj)
  return λ * norm(y .+ ψ.xk .+ ψ.sj, Inf)
end

function _proj_l1ball(v::AbstractVector{R}, r::R) where {R <: Real}
  # Implements algorithm proposed in:
  # Duchi et al. "Efficient Projections onto the ℓ₁-ball for Learning in High Dimensions",
  if norm(v, 1) ≤ r
    return copy(v)
  end
  μ = sort(abs.(v), rev = true)
  cssμ = cumsum(μ)
  list_inx = collect(1:length(μ))
  rho = findlast(μ .* list_inx - (cssμ .- r) .+ eps(R) .> 0)
  θ = (cssμ[rho] - r) / rho
  return sign.(v) .* max.(abs.(v) .- θ, zero(R))
end