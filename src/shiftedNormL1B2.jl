export ShiftedNormL1B2

mutable struct ShiftedNormL1B2{R <: Real, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} <: ShiftedProximableFunction
  h::NormL1{R}
  x::V1
  s::V2
  Δ::R
  χ::NormL2{R}
  function ShiftedNormL1B2(h::NormL1{R}, x::AbstractVector{R}, Δ::R, χ::NormL2{R}) where {R <: Real}
    s = similar(x)
    new{R, typeof(x), typeof(s)}(h, x, s, Δ, χ)
  end
end

shifted(h::NormL1{R}, x::AbstractVector{R}, Δ::R, χ::NormL2{R}) where {R <: Real} = ShiftedNormL1B2(h, x, Δ, χ)

fun_name(ψ::ShiftedNormL1B2) = "shifted L1 norm with L₂-norm trust region indicator"
fun_expr(ψ::ShiftedNormL1B2) = "s ↦ h(x + s) + χ({‖s‖₂ ≤ Δ})"
fun_params(ψ::ShiftedNormL1B2) = "x = $(ψ.x), Δ = $(ψ.Δ)"

function prox(ψ::ShiftedNormL1B2{R, V1, V2}, q::AbstractVector{R}, σ::R) where {R <: Real, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}

  ProjB(y) = min.(max.(y, q .- σ), q .+ σ)
  froot(η) = η - norm(ProjB((-ψ.x) .* (η / ψ.Δ)))

  ψ.s .= ProjB(-ψ.x)

  if norm(ψ.s) > ψ.Δ
    η = fzero(froot, 1e-10, Inf)
    ψ.s .= ProjB(-ψ.x) .* (η / ψ.Δ)
  end
  if norm(ψ.s) > ψ.Δ
    ψ.s .=*(ψ.Δ /norm(ψ.s))
  end
  return ψ.s
end