export ShiftedNormL2BInf

mutable struct ShiftedNormL2BInf{
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
  X::Conjugate{IndBallL1{R}}
  shifted_twice::Bool

  function ShiftedNormL2BInf(
    h::NormL2{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    Δ::R,
    X::Conjugate{IndBallL1{R}},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(sj)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, Δ, X, shifted_twice)
  end
end

(ψ::ShiftedNormL2BInf)(y) = ψ.h(ψ.xk + ψ.sj + y) + IndBallLinf(ψ.Δ)(ψ.sj + y)

shifted(h::NormL2{R}, xk::AbstractVector{R}, Δ::R, X::Conjugate{IndBallL1{R}}) where {R <: Real} =
  ShiftedNormL2BInf(h, xk, zero(xk), Δ, X, false)
shifted(
  ψ::ShiftedNormL2BInf{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedNormL2BInf(ψ.h, ψ.xk, sj, ψ.Δ, ψ.X, true)

fun_name(ψ::ShiftedNormL2BInf) = "shifted L2 norm with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedNormL2BInf) = "t ↦ ‖xk + sj + t‖2 + X({‖sj + t‖∞ ≤ Δ})"
fun_params(ψ::ShiftedNormL2BInf) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "Δ = $(ψ.Δ)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL2BInf{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}

  ψ.sol .= q + ψ.xk + ψ.sj
  ## case 1
  w = zeros(y)
  for i = 1:numel(w)
    if abs(ψ.xk[i]) < ψ.Δ || ψ.xk[i]*ψ.sol[i] > 0
      w[i] = 0
    elseif (- ψ.xk[i] == ψ.Δ && ψ.sol[i] > 0) || (ψ.xk[i] == ψ.Δ && ψ.sol[i] < 0)
      w[i] = ψ.sol[i] / σ
    end
  end
  ## check to see if satisfied
  if sqrt(sum((w - ψ.sol./σ).^2)) < 1 && ψ.X(ψ.xk) <= ψ.Δ
    snorm = sqrt(sum((ψ.sol - σ .* w).^2))
    y .= max(0, σ/snorm) .* (ψ.sol - σ .* w) - (ψ.xk + ψ.sj)
    return y
  end

  ## case 2
  softthres(x, a) = sign(x) .* max(0, abs.(x) .- a)
  froot(n) = n - sqrt(sum((σ .* softthres((ψ.sol - (n/(n - σ)) .* ψ.xk)./σ, (ψ.Δ/σ*(n/(n - σ))) - ψ.sol )).^2))
  n = find_zero(froot, ψ.Δ)
  w .= softthres((ψ.sol - n/(n - σ) .* ψ.xk)./σ, ψ.Δ/σ*(n/(n - σ)))
  y .= max(0, σ/sqrt(sum((ψ.sol - σ .* w).^2))) .* (ψ.sol - σ .* w)  - (ψ.xk + ψ.sj)

  return y
end
