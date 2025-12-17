export ShiftedGroupNormL2Binf

mutable struct ShiftedGroupNormL2Binf{
  R <: Real,
  RR <: AbstractVector{R},
  I,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::GroupNormL2{R, RR, I}
  xk::V0
  sj::V1
  sol::V2
  Δ::R
  χ::Conjugate{IndBallL1{R}}
  shifted_twice::Bool
  xsy::V2

  function ShiftedGroupNormL2Binf(
    h::GroupNormL2{R, RR, I},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    Δ::R,
    χ::Conjugate{IndBallL1{R}},
    shifted_twice::Bool,
  ) where {R <: Real, RR <: AbstractVector{R}, I}
    sol = similar(sj)
    xsy = similar(sj)
    new{R, RR, I, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, Δ, χ, shifted_twice, xsy)
  end
end

function (ψ::ShiftedGroupNormL2Binf)(y)
  @. ψ.xsy = ψ.sj + y
  indball_val = IndBallLinf(1.1 * ψ.Δ)(ψ.xsy)
  ψ.xsy .+= ψ.xk
  return ψ.h(ψ.xsy) + indball_val
end

shifted(
  ψ::ShiftedGroupNormL2Binf{R, RR, I, V0, V1, V2},
  sj::AbstractVector{R},
) where {
  R <: Real,
  RR <: AbstractVector{R},
  I,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} = ShiftedGroupNormL2Binf(ψ.h, ψ.xk, sj, ψ.Δ, ψ.χ, true)

fun_name(ψ::ShiftedGroupNormL2Binf) = "shifted ∑ᵢ‖⋅‖₂ norm with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedGroupNormL2Binf) = "t ↦ ∑ᵢ ‖xk + sj + t‖₂ +  X({‖sj + t‖∞ ≤ Δ})"
fun_params(ψ::ShiftedGroupNormL2Binf) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "Δ = $(ψ.Δ)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedGroupNormL2Binf{R, RR, I, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {
  R <: Real,
  RR <: AbstractVector{R},
  I,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
}
  ψ.sol .= q .+ ψ.xk .+ ψ.sj
  ϵ = 1 ## sasha's initial guess

  for (idx, λ) ∈ zip(ψ.h.idx, ψ.h.lambda)
    σλ = λ * σ

    # Views for block data
    @views begin
      solb = ψ.sol[idx]
      xkb = ψ.xk[idx]
      sjb = ψ.sj[idx]
      tmpb = ψ.xsy[1:length(solb)]
    end

    # in-place soft threshold into tmpb: tmpb .= sign.(expr) .* max.(0, abs.(expr) .- a)
    function softthres_block!(dest, a, nfactor)
      @inbounds for i in eachindex(dest)
        val = solb[i] / σ - nfactor * xkb[i]
        dv = abs(val) - a
        dest[i] = dv > 0 ? sign(val) * dv : zero(eltype(dest))
      end
    end

    # compute froot using in-place operations
    function froot(n)
      nfac = n / (σ * (n - σλ))
      ath = ψ.Δ * nfac
      softthres_block!(tmpb, ath, nfac)
      # tmpb currently holds softthres(expr, ath)
      @inbounds begin
        # compute tmpb .-= solb  (in-place)
        s = zero(eltype(tmpb))
        for i in eachindex(tmpb)
          tmpb[i] -= solb[i]
          s += tmpb[i]^2
        end
        return n - sqrt(s)
      end
    end

    lmin = σλ * (1 + eps(R)) # lower bound
    fl = froot(lmin)

    ansatz = lmin + ϵ #ansatz for upper bound
    step = ansatz / (σ * (ansatz - σλ))
    # compute zlmax using in-place softthres
    softthres_block!(ψ.xsy[1:length(solb)], ψ.Δ * step, step)
    zlmax = 0.0
    @inbounds for i in 1:length(solb)
      zlmax += ψ.xsy[i]^2
    end
    zlmax = sqrt(zlmax)

    lmax = norm(solb) + σ * (zlmax + abs((ϵ - 1) / ϵ + 1) * λ * norm(xkb))
    fm = froot(lmax)
    if fl * fm > 0
      @inbounds for i in eachindex(idx)
        y[idx[i]] = zero(eltype(y))
      end
    else
      n = fzero(froot, lmin, lmax)
      step = n / (σ * (n - σλ))
      if abs(n - σλ) ≈ 0
        @inbounds for i in eachindex(idx)
          y[idx[i]] = zero(eltype(y))
        end
      else
        # compute solb .- σ .* softthres(... ) into tmpb
        nfac = step
        ath = ψ.Δ * nfac
        @inbounds for i in eachindex(solb)
          val = solb[i] / σ - nfac * xkb[i]
          dv = abs(val) - ath
          tmpb[i] = dv > 0 ? sign(val) * dv : zero(eltype(tmpb))
        end
        @inbounds for i in eachindex(tmpb)
          tmpb[i] = solb[i] - σ * tmpb[i]
        end
        # apply l2prox in-place into y[idx]
        s = zero(eltype(tmpb))
        @inbounds for i in eachindex(tmpb)
          s += tmpb[i]^2
        end
        s = sqrt(s)
        factor = s == 0 ? zero(eltype(s)) : max(0, 1 - σλ / s)
        @inbounds for i in eachindex(tmpb)
          y[idx[i]] = factor * tmpb[i]
        end
      end
    end
    # subtract shifts in-place
    @inbounds for (k, gi) in enumerate(idx)
      y[gi] -= (ψ.xk[gi] + ψ.sj[gi])
    end
  end
  return y
end
