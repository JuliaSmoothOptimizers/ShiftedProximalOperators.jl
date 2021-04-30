export ShiftedNormL1BInf

mutable struct ShiftedNormL1BInf{R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} <: ShiftedProximableFunction
  h::NormL1{R}
  x0::V0
  x::V1
  s::V2
  Δ::R
  χ::Conjugate{IndBallL1{R}}

  function ShiftedNormL1BInf(h::NormL1{R}, x0::AbstractVector{R}, x::AbstractVector{R}, Δ::R, χ::Conjugate{IndBallL1{R}}) where {R <: Real}
    s = similar(x)
    new{R, typeof(x0), typeof(x), typeof(s)}(h, x0, x, s, Δ, χ)
  end
end

shifted(h::NormL1{R}, x::AbstractVector{R}, Δ::R, χ::Conjugate{IndBallL1{R}}) where {R <: Real} = ShiftedNormL1BInf(h, zero(x), x, Δ, χ)
shifted(ψ::ShiftedNormL1BInf{R, V0, V1, V2}, x::AbstractVector{R}) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} = ShiftedNormL1BInf(ψ.h, ψ.x, x, ψ.Δ, ψ.χ)

fun_name(ψ::ShiftedNormL1BInf) = "shifted L1 norm with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedNormL1BInf) = "s ↦ ‖x + s‖₁ + χ({‖s‖∞ ≤ Δ})"
fun_params(ψ::ShiftedNormL1BInf) = "x0 = $(ψ.x0)\n" * " "^14 * "x = $(ψ.x), Δ = $(ψ.Δ)"

function prox(ψ::ShiftedNormL1BInf{R, V0, V1, V2}, q::AbstractVector{R}, σ::R) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  ψ.s .= -ψ.x .- ψ.x0

  for i ∈ eachindex(ψ.s)
    ψ.s[i] = min(max(min(max(ψ.s[i], q[i] - ψ.λ * σ), q[i] + ψ.λ * σ), -ψ.Δ), ψ.Δ)
  end

  return ψ.s
end
