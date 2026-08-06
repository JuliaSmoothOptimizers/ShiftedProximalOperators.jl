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
  indball_val = IndBallLinf(1.01 * ψ.Δ)(ψ.xsy)
  ψ.xsy .+= ψ.xk
  return ψ.h(ψ.xsy) + indball_val
end

shifted(
  h::GroupNormL2{R, RR, I},
  xk::AbstractVector{R},
  Δ::R,
  χ::Conjugate{IndBallL1{R}},
) where {R <: Real, RR <: AbstractVector{R}, I} =
  ShiftedGroupNormL2Binf(h, xk, zero(xk), Δ, χ, false)
shifted(h::NormL2{R}, xk::AbstractVector{R}, Δ::R, χ::Conjugate{IndBallL1{R}}) where {R <: Real} =
  ShiftedGroupNormL2Binf(GroupNormL2([h.lambda]), xk, zero(xk), Δ, χ, false)
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

  softthres(x, a) = sign.(x) .* max.(0, abs.(x) .- a)
  l2prox(x, a) = max(0, 1 - a / norm(x)) .* x
  linfproj(x) = max.(-ψ.Δ, min.(ψ.Δ, x))
  for (idx, λ) ∈ zip(ψ.h.idx, ψ.h.lambda)
    σλ = λ * σ
    froot(n) =
      n - norm(
        (n / (n - σλ)) .* (
          ψ.xk[idx] .+ linfproj(
            ((n - σλ)/n) .* ψ.sol[idx] .- ψ.xk[idx]
          )
        )
      )
    xlength = length(ψ.xk[idx])
    xnorminf = ψ.χ(ψ.xk[idx])
    xnorm = norm(ψ.xk[idx])
    qnorm = norm(ψ.sol[idx])
    qproj = sum(linfproj(ψ.sol[idx] .- ψ.xk[idx]) - ψ.xk[idx])
    τ = 1e4*eps(R)
    n = NaN
    if xnorminf > ψ.Δ #case 1
      n = find_zero(froot, (σλ + xnorminf  - ψ.Δ, σλ + xnorm + ψ.Δ*√xlength), Roots.A42())
    elseif qnorm > σλ &&  xnorminf < ψ.Δ #case 2a
      n = find_zero(froot, (σλ + τ, qnorm + xnorm + ψ.Δ*√xlength), Roots.A42())
    elseif qnorm > σλ && xnorm ≈ ψ.Δ && qproj ≈ 0.0 #case 4a
      n = find_zero(froot, (σλ + τ, min(σλ + xnorm + ψ.Δ*sqrt(n), qnorm)), Roots.A42())
    end
    if isnan(n)
      y[idx] .= 0
    else
      step = n / (σ * (n - σλ))
      y[idx] .= l2prox(
        ψ.sol[idx] .- σ .* softthres((ψ.sol[idx] ./ σ .- step .* ψ.xk[idx]), ψ.Δ * step),
        σλ,
      )
    end
    y[idx] .-= (ψ.xk[idx] + ψ.sj[idx])
  end
  return y
end
