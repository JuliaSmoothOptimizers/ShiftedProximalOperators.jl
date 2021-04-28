export ShiftedNormL1B2

mutable struct ShiftedNormL1B2{R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} <: ShiftedProximableFunction
  h::NormL1{R}
  x0::V0
  x::V1
  s::V2
  Δ::R
  χ::NormL2{R}
  function ShiftedNormL1B2(h::NormL1{R}, x0::AbstractVector{R}, x::AbstractVector{R}, Δ::R, χ::NormL2{R}) where {R <: Real}
    s = similar(x)
    new{R, typeof(x0), typeof(x), typeof(s)}(h, x0, x, s, Δ, χ)
  end
end

shifted(h::NormL1{R}, x::AbstractVector{R}, Δ::R, χ::NormL2{R}) where {R <: Real} = ShiftedNormL1B2(h, zero(x), x, Δ, χ)
shifted(ψ::ShiftedNormL1B2{R, V0, V1, V2}, x::AbstractVector{R}, Δ::R, χ::NormL2{R}) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} = ShiftedNormL1B2(ψ.h, ψ.x, x, Δ, χ)

fun_name(ψ::ShiftedNormL1B2) = "shifted L1 norm with L₂-norm trust region indicator"
fun_expr(ψ::ShiftedNormL1B2) = "s ↦ h(x + s) + χ({‖s‖₂ ≤ Δ})"
fun_params(ψ::ShiftedNormL1B2) = "x = $(ψ.x), Δ = $(ψ.Δ)"

function prox(ψ::ShiftedNormL1B2{R, V0, V1, V2}, q::AbstractVector{R}, σ::R) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}

  ProjB(y) = min.(max.(y, q .- ψ.λ * σ), q .+ ψ.λ * σ)
  froot(η) = η - ψ.χ(ProjB((-ψ.x - ψ.x0) .* (η / ψ.Δ)))

  ψ.s .= ProjB(-ψ.x - ψ.x0)

  if ψ.χ(ψ.s) > ψ.Δ
    η = fzero(froot, 1e-10, Inf)
    ψ.s .=* (η / ψ.Δ)
  end
  if ψ.χ(ψ.s) > ψ.Δ
    ψ.s .=* (ψ.Δ / ψ.χ(ψ.s))
  end
  return ψ.s
end