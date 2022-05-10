export ShiftedNormL1BInf

mutable struct ShiftedNormL1BInf{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3 <: AbstractVector{R}
} <: ShiftedProximableFunction
  h::NormL1{R}
  xk::V0
  sj::V1
  sol::V2
  l::V3
  u::V3
  shifted_twice::Bool

  function ShiftedNormL1BInf(
    h::NormL1{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    l::AbstractVector{R},
    u::AbstractVector{R},
    shifted_twice::Bool,
  ) where {R <: Real}
    sol = similar(xk)
    if sum(l .> u) > 0 
      error("Error on the trust region bounds, at least one lower bound is greater than the upper bound.")
    else
      new{R, typeof(xk), typeof(sj), typeof(sol), typeof(l)}(h, xk, sj, sol, l, u, shifted_twice)
    end
  end
end

# We cannot use this function anymore with [l,u] trust region 
#=
(ψ::ShiftedNormL1BInf)(y) = ψ.h(ψ.xk + ψ.sj + y) + IndBallLinf(ψ.Δ)(ψ.sj + y)
=#

shifted(h::NormL1{R}, xk::AbstractVector{R}, l::AbstractVector{R}, u::AbstractVector{R}) where {R <: Real} =
  ShiftedNormL1BInf(h, xk, zero(xk), l, u, false)
shifted(
  ψ::ShiftedNormL1BInf{R, V0, V1, V2, V3},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3 <: AbstractVector{R}} =
  ShiftedNormL1BInf(ψ.h, ψ.xk, sj, ψ.l, ψ.u, true)

fun_name(ψ::ShiftedNormL1BInf) = "shifted L1 norm with generalized trust region indicator"
fun_expr(ψ::ShiftedNormL1BInf) = "t ↦ ‖xk + sj + t‖₁ + χ({sj + t .∈ [l,u]})"
fun_params(ψ::ShiftedNormL1BInf) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "l = $(ψ.l)\n" * " "^14 * "u = $(ψ.u)"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL1BInf{R, V0, V1, V2, V3},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3 <: AbstractVector{R}}
  
  σλ = σ * ψ.λ

  for i ∈ eachindex(y)

    li = ψ.l[i]
    ui = ψ.u[i]
    qi = q[i]
    xs = ψ.xk[i] + ψ.sj[i]

    if ui <= -xs 

      candidates = [li, ui, qi + σλ]
      Σi = candidates[li .<= candidates .<= ui] # set of potential solutions
      y[i] = Σi[argmin((Σi .- qi).^2 + 2 * σλ .* abs.(xs .+ Σi))]

    elseif -xs <= li

      candidates = [li, ui, qi - σλ]
      Σi = candidates[li .<= candidates .<= ui] # set of potential solutions
      y[i] = Σi[argmin((Σi .- qi).^2 + 2 * σλ .* abs.(xs .+ Σi))]

    else 

      candidates1 = [li, -xs, qi + σλ]
      Σi1 = candidates1[li .<= candidates1 .<= -xs] # set of potential "left" solutions

      candidates2 = [-xs, ui, qi - σλ]
      Σi2 = candidates2[-xs .<= candidates2 .<= ui] # set of potential "right" solutions

      Σi = vcat(Σi1, Σi2)
      y[i] = Σi[argmin((Σi .- qi).^2 + 2 * σλ .* abs.(xs .+ Σi))]

    end

  end

  return y
end
