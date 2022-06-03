export ShiftedNormL2Binf

mutable struct ShiftedNormL2Binf{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::NormL2{R}
  xk::V0
  sj::V1
  sol::V2
  Δ::R
  χ::Conjugate{IndBallL1{R}}
  shifted_twice::Bool

  function ShiftedNormL2Binf(
    h::NormL2{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    Δ::R,
    χ::Conjugate{IndBallL1{R}},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(sj)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, Δ,  χ, shifted_twice)
  end
end

(ψ::ShiftedNormL2Binf)(y) = ψ.h(ψ.xk + ψ.sj + y) + IndBallLinf(ψ.Δ)(ψ.sj + y)

shifted(h::NormL2{R}, xk::AbstractVector{R}, Δ::R,  χ::Conjugate{IndBallL1{R}}) where {R <: Real} =
  ShiftedNormL2Binf(h, xk, zero(xk), Δ,  χ, false)
shifted(
  ψ::ShiftedNormL2Binf{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormL2Binf(ψ.h, ψ.xk, sj, ψ.Δ, ψ.χ, true)

fun_name(ψ::ShiftedNormL2Binf) = "shifted L2 norm with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedNormL2Binf) = "t ↦ ‖xk + sj + t‖2 + X({‖sj + t‖∞ ≤ Δ})"
fun_params(ψ::ShiftedNormL2Binf) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "Δ = $(ψ.Δ)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL2Binf{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}

  ψ.sol .= q + ψ.xk + ψ.sj
	w = zeros(size(q)) #change eventually
  softthres(x, a) = sign.(x) .* max.(0, abs.(x) .- a)
	l2prox(x,a) = max(0, 1- a/sqrt(sum(x.^2))).*x
	froot(n) = n - sqrt(sum((σ .* softthres((ψ.sol./σ .- (n/(σ*(n - σ))) .* ψ.xk), ψ.Δ*(n/(σ*(n - σ)))) .- ψ.sol ).^2))
  
	lmin = σ + 1e-10
	lmax = maximum(abs.(ψ.sol))/σ + 1e4*maximum(abs.(ψ.xk))/min(σ,1)
	fl = froot(lmin)
	fm = froot(lmax)

	if fl*fm > 0
		n  = 0
		y .= 0
	else
	  n = fzero(froot, lmin, lmax)
		step = n / (σ*( n - σ))
    w .= softthres((ψ.sol./σ .- step .* ψ.xk), ψ.Δ*step)
		y .= l2prox(ψ.sol .- σ.*w, σ)
  end
	y .-= (ψ.xk + ψ.sj)
  return y
end
