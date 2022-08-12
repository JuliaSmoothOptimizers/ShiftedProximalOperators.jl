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

  function ShiftedGroupNormL2Binf(
    h::GroupNormL2{R, RR, I},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    Δ::R,
    χ::Conjugate{IndBallL1{R}},
    shifted_twice::Bool,
  ) where {R <: Real, RR <: AbstractVector{R}, I}
    sol = similar(sj)
    new{R, RR, I, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, Δ, χ, shifted_twice)
  end
end

(ψ::ShiftedGroupNormL2Binf)(y) = ψ.h(ψ.xk + ψ.sj + y) .+ IndBallLinf(1.1 * ψ.Δ)(ψ.sj .+ y)

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
  ϵ = 1 ## sasha's initial guess
  softthres(x, a) = sign.(x) .* max.(0, abs.(x) .- a)
  l2prox(x, a) = max(0, 1 - a / norm(x)) .* x
  for (idx, λ) ∈ zip(ψ.h.idx, ψ.h.lambda)
    σλ = λ * σ
    ## find root for each block
    froot(n) =
      n - norm(
        σ .* softthres(
          (ψ.sol[idx] ./ σ .- (n / (σ * (n - σλ))) .* ψ.xk[idx]),
          ψ.Δ * (n / (σ * (n - σλ))),
        ) .- ψ.sol[idx],
      )
    lmin = σλ * (1 + eps(R)) # lower bound
    fl = froot(lmin)

    ansatz = lmin + ϵ #ansatz for upper bound
    step = ansatz / (σ * (ansatz - σλ))
    zlmax = norm(softthres((ψ.sol[idx] ./ σ .- step .* ψ.xk[idx]), ψ.Δ * step))
    lmax = norm(ψ.sol[idx]) + σ * (zlmax + abs((ϵ - 1) / ϵ + 1) * λ * norm(ψ.xk[idx]))
    fm = froot(lmax)
    if fl * fm > 0
      y[idx] .= 0
    else
      n = fzero(froot, lmin, lmax)
      step = n / (σ * (n - σλ))
      if abs(n - σλ) ≈ 0
        y[idx] .= 0
      else
        y[idx] .= l2prox(
          ψ.sol[idx] .- σ .* softthres((ψ.sol[idx] ./ σ .- step .* ψ.xk[idx]), ψ.Δ * step),
          σλ,
        )
      end
    end
    y[idx] .-= (ψ.xk[idx] + ψ.sj[idx])
  end
  return y
end
