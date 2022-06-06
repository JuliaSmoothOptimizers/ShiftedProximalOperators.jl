export ShiftedNormL2GroupBinf

mutable struct ShiftedNormL2GroupBinf{
  R  <: Real,
	RR <: AbstractVector{R},
  I  <: Vector{Vector{Int}},
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::GroupNormL2{R,RR,I}
  xk::V0
  sj::V1
  sol::V2
  Δ::R
  χ::Conjugate{IndBallL1{R}}
  shifted_twice::Bool

  function ShiftedNormL2GroupBinf(
    h::GroupNormL2{R,RR,I},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    Δ::R,
    χ::Conjugate{IndBallL1{R}},
    shifted_twice::Bool,
		) where {R <: Real, RR <: AbstractVector{R}, I <: Vector{Vector{Int}}}
    sol = similar(sj)
    new{R,RR,I,typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, Δ, χ, shifted_twice)
  end
end

(ψ::ShiftedNormL2GroupBinf)(y) = ψ.h(ψ.xk + ψ.sj + y) .+ IndBallLinf(1.1*ψ.Δ)(ψ.sj + y) # ".+" fixes Float32 error in runtests.

shifted(
  h::GroupNormL2{R,RR,I},
  xk::AbstractVector{R},
  Δ::R,
  χ::Conjugate{IndBallL1{R}}
 ) where {R <: Real, RR <: AbstractVector{R}, I <: Vector{Vector{Int}}} = ShiftedNormL2GroupBinf(h, xk, zero(xk), Δ, χ,  false)
shifted(
  ψ::ShiftedNormL2GroupBinf{R, RR, I, V0, V1, V2},
  sj::AbstractVector{R},
 ) where {R <: Real, RR <: AbstractVector{R}, I <: Vector{Vector{Int}}, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
ShiftedNormL2GroupBinf(ψ.h, ψ.xk, sj, ψ.Δ, ψ.χ, true)

fun_name(ψ::ShiftedNormL2GroupBinf) = "shifte ∑ᵢ||⋅||_2 norm with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedNormL2GroupBinf) = "t ↦ ‖xk + sj + t‖ₚᵖ, p = 2 +  X({‖sj + t‖∞ ≤ Δ})"
fun_params(ψ::ShiftedNormL2GroupBinf) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "Δ = $(ψ.Δ)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL2GroupBinf{R, RR,  I, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
 ) where {R <: Real, RR <: AbstractVector{R}, I<: Vector{Vector{Int}}, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}

  ψ.sol .= q + ψ.xk + ψ.sj
	w = zeros(size(y)) #### change eventually to preallocate
	softthres(x, a) = sign.(x) .* max.(0, abs.(x) .- a)
  l2prox(x,a) = max(0, 1- a/sqrt(sum(x.^2))).*x
  for i = 1:length(ψ.h.idx)
    idx = ψ.h.idx[i]
    xk  = ψ.xk[idx]
    sj  = ψ.sj[idx]
    sol = ψ.sol[idx]
    σλ  = ψ.h.lambda[i]*σ
    ## find root for each block
    froot(n) = n - sqrt(sum((σλ .* softthres((sol./σλ .- (n/(σλ*(n - σλ))) .* xk), ψ.Δ*(n/(σλ*(n - σλ)))) .- sol ).^2))

		lmin = σλ + 1e-10
 	 	# lmax = maximum(abs.(sol))/σλ + 1e4*maximum(abs.(xk))/min(σλ,1) #do these once?
    lmax = min(1, σλ)*1e4
 	 	fl = froot(lmin)
 	 	fm = froot(lmax)
 	 	if fl*fm > 0
			y[idx] .= 0
 	 	else
 	 	  n = fzero(froot, lmin, lmax)
 	 	  step = n / (σλ*(n - σλ))
      if abs(n - σλ) ≈ 0
        y[idx] .= 0
      else
        w[idx] .= softthres((sol./σλ .- step .* xk), ψ.Δ*step)
        y[idx] .= l2prox(sol .- σλ.*w[idx], σλ)
      end
 	 	end
		y[idx] .-= (xk + sj)
	end
  # @show q, σ, ψ.h.idx
 	return y
end




