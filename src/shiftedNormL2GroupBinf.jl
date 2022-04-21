export ShiftedNormL2GroupBinf

mutable struct ShiftedNormL2GroupBinf{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::GroupNormL2{R}
  xk::V0
  sj::V1
  sol::V2
  Δ::R
  X::Conjugate{IndBallL1{R}}
  shifted_twice::Bool

  function ShiftedNormL2GroupBinf(
    h::GroupNormL2{R},
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

(ψ::ShiftedNormL2GroupBinf)(y) = ψ.h(ψ.xk + ψ.sj + y) + IndBallLinf(ψ.Δ)(ψ.sj + y)

shifted(
  h::GroupNormL2{R},
  xk::AbstractVector{R},
  Δ::R,
  X::Conjugate{IndBallL1{R}}
) where {R <: Real} = ShiftedNormL2GroupBinf(h, xk, zero(xk), Δ, X,  false)
shifted(
  ψ::ShiftedNormL2GroupBinf{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
ShiftedNormL2GroupBinf(ψ.h, ψ.xk, sj, ψ.Δ, ψ.X, true)

fun_name(ψ::ShiftedNormL2GroupBinf) = "shifte ∑ᵢ||⋅||_2 norm with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedNormL2GroupBinf) = "t ↦ ‖xk + sj + t‖ₚᵖ, p = 2 +  X({‖sj + t‖∞ ≤ Δ})"
fun_params(ψ::ShiftedNormL2GroupBinf) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "Δ = $(ψ.Δ)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL2GroupBinf{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  
  ψ.sol .= q + ψ.xk + ψ.sj
  w = zeros(y)
  
  c1true = 0
  for i = 1:ψ.h.g
    ## case 1
    idx  = ψ.h.idx[1,i]:ψ.h.idx[2,i]
    temp = w[idx]
    xk   = ψ.xk[idx]
    sol  = ψ.sol[idx]
    for i = numel(temp)
      if abs(temp[i]) < ψ.Δ || xk[i]*sol[i] > 0
        w[i] = 0
      else
        w[i] = sol[i] / σ
      end
    end
    w[idx] = temp

    ## check condition
    if norm(w[idx] - sol ./ σ) < 1 && ψ.X(xk) <= ψ.Δ
      y[idx] = max(0, σ/norm(sol - σ .* w[idx])) .* (sol - σ .* w[idx]) - (xk + sj)
      c1true += 1
    end
  end
  
  if c1true == ψ.h.g
    return y
  end

  softthres(x, a) = sign(x) .* max(0, abs.(x) .- a)

  for i = 1:ψ.h.g
    idx = ψ.h.idx[1,i]:ψ.h.idx[2,i]
    xk  = ψ.xk[idx]
    sj  = ψ.sj[idx]
    sol = ψ.sol[idx]
    ## find root for each block
    froot(n) = n - norm(σ .* softthres((sol - (n/(n - σ)) .* xk)./σ, (ψ.Δ/σ*(n/(n - σ))) - sol))
    n = find_zero(froot, ψ.Δ)
    temp = w[idx]
    w[idx] .= softthres((sol - n/(n - σ) .* xk)./σ, ψ.Δ/σ*(n/(n - σ)))
    y[idx] .= max(0, σ/norm(sol - σ .* w)) .* (sol - σ .* temp)  - (xk + sj)


  #for i = 1:ψ.h.g
  #  y[ψ.h.idx[1,i]:ψ.h.idx[2,i]] .= max(1 - σ/norm(ψ.sol[ψ.h.idx[1,i]:ψ.h.idx[2,i]]), 0) .* ψ.sol[ψ.h.idx[1,i]:ψ.h.idx[2,i]]
  #end
  #y .-= (ψ.xk + ψ.sj)
  end
 
  return y
end




