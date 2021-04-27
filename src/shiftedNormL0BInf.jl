export ShiftedNormL0BInf

mutable struct ShiftedNormL0BInf{R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} <: ShiftedProximableFunction
  h::NormL0{R}
  x0::V0
  x::V1
  s::V2
  Δ::R
  χ::Conjugate{IndBallL1{R}}
  function ShiftedNormL0BInf(h::NormL0{R}, x0::AbstractVector{R}, x::AbstractVector{R}, Δ::R,  χ::Conjugate{IndBallL1{R}}) where {R <: Real}
    s = similar(x)
    new{R, typeof(x), typeof(s)}(h, x0, x, s, Δ, χ)
  end
end

shifted(h::NormL0{R}, x::AbstractVector{R}, Δ::R, χ::Conjugate{IndBallL1{R}}) where {R <: Real} = ShiftedNormL0BInf(h, zero(x), x, Δ, χ)
shifted(ψ::ShiftedNormL0BInf{R, V0, V1, V2}, x::AbstractVector{R}, Δ::R, χ::Conjugate{IndBallL0{R}}) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R} } = ShiftedNormL0BInf(ψ.h, ψ.x, x, Δ, χ)

fun_name(ψ::ShiftedNormL0BInf) = "shifted L0 pseudo-norm with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedNormL0BInf) = "s ↦ h(x + s) + χ({‖s‖∞ ≤ Δ})"
fun_params(ψ::ShiftedNormL0BInf) = "x = $(ψ.x), Δ = $(ψ.Δ)"

function prox(ψ::ShiftedNormL0BInf{R, V0, V1, V2}, q::AbstractVector{R}, σ::R) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  ProjB!(y) = begin
    for i ∈ eachindex(y)
      y[i] = min(max(y[i], (ψ.x[i] + ψ.x0[i]) - ψ.Δ), (ψ.x[i] + ψ.x0[i]) + ψ.Δ)
    end
  end 
  c = sqrt(2 * ψ.λ * σ)

  for i ∈ eachindex(q)
    xpq = (ψ.x[i] + ψ.x0[i]) + q[i]
    if abs(xpq) ≤ c
      ψ.s[i] = 0
    else
      ψ.s[i] = xpq
    end
  end
  ProjB!(ψ.s) 
  ψ.s .-= (ψ.x + ψ.x0)
  return ψ.s 
end
