export ShiftedNormL0BInf

mutable struct ShiftedNormL0BInf{R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} <: ShiftedProximableFunction
  h::NormL0{R}
  x0::V0  # base shift (nonzero when shifting an already shifted function
  x::V1
  s::V2
  Δ::R
  χ::Conjugate{IndBallL1{R}}
  function ShiftedNormL0BInf(h::NormL0{R}, x0::AbstractVector{R}, x::AbstractVector{R}, Δ::R,  χ::Conjugate{IndBallL1{R}}) where {R <: Real}
    s = similar(x)
    new{R, typeof(x0), typeof(x), typeof(s)}(h, x0, x, s, Δ, χ)
  end
end

shifted(h::NormL0{R}, x::AbstractVector{R}, Δ::R, χ::Conjugate{IndBallL1{R}}) where {R <: Real} = ShiftedNormL0BInf(h, zero(x), x, Δ, χ)
shifted(ψ::ShiftedNormL0BInf{R, V0, V1, V2}, x::AbstractVector{R}, Δ::R, χ::Conjugate{IndBallL0{R}}) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R} } = ShiftedNormL0BInf(ψ.h, ψ.x, x, Δ, χ)

fun_name(ψ::ShiftedNormL0BInf) = "shifted L0 pseudo-norm with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedNormL0BInf) = "s ↦ h(x + s) + χ({‖s‖∞ ≤ Δ})"
fun_params(ψ::ShiftedNormL0BInf) = "x0 = $(ψ.x0)\n" * " "^14 * "x = $(ψ.x), Δ = $(ψ.Δ)"

function prox(ψ::ShiftedNormL0BInf{R, V0, V1, V2}, q::AbstractVector{R}, σ::R) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  c = sqrt(2 * ψ.λ * σ)

  for i ∈ eachindex(q)
    xpq = ψ.x0[i] + ψ.x[i] + q[i]
    if abs(xpq) ≤ c
      ψ.s[i] = 0
    else
      ψ.s[i] = xpq
    end
  end

  for i ∈ eachindex(ψ.s)
    ψ.s[i] = min(max(ψ.s[i], ψ.x0[i] + ψ.x[i] - ψ.Δ), ψ.x0[i] + ψ.x[i] + ψ.Δ)
  end

  ψ.s .-= ψ.x0 .+ ψ.x
  return ψ.s
end
