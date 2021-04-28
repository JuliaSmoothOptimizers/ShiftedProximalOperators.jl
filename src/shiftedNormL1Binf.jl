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
shifted(ψ::ShiftedNormL1BInf{R, V0, V1, V2}, x::AbstractVector{R}, Δ::R, χ::NormL2{R}) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} = ShiftedNormL1BInf(ψ.h, ψ.x, x, Δ, χ)


fun_name(ψ::ShiftedNormL1BInf) = "shifted L1 norm with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedNormL1BInf) = "s ↦ h(x + s) + χ({‖s‖∞ ≤ Δ})"
fun_params(ψ::ShiftedNormL1BInf) = "x = $(ψ.x), Δ = $(ψ.Δ)"

function prox(ψ::ShiftedNormL1BInf{R, V0, V1, V2}, q::AbstractVector{R}, σ::R) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  ProjB!(w) = begin 
    for i ∈ eachindex(w)
      w[i] = min(max(w[i], q[i] - ψ.λ * σ), q[i] + ψ.λ * σ)
    end
  end
  ProjΔ!(y) = begin 
    for i ∈ eachindex(y)
      y[i] = min(max(y[i], - ψ.Δ), ψ.Δ)
    end
  end
  ψ.s .= -ψ.x - ψ.x0
  @show ProjB!(ψ.s)
  ProjΔ!(ProjB!(ψ.s))
  return ψ.s
end