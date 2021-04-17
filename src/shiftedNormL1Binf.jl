export ShiftedNormL1BInf

mutable struct ShiftedNormL1BInf{R <: Real, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} <: ShiftedProximableFunction
  h::NormL1{R}
  x::V1
  s::V2
  Δ::R
  function ShiftedNormL1BInf(h::NormL1{R}, x::AbstractVector{R}, Δ::R) where {R <: Real}
    s = similar(x)
    new{R, typeof(x), typeof(s)}(h, x, s, Δ)
  end
end

shifted(h::NormL1{R}, x::AbstractVector{R}, Δ::R) where {R <: Real} = ShiftedNormL1BInf(h, x, Δ)

fun_name(ψ::ShiftedNormL1BInf) = "shifted L1 norm with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedNormL1BInf) = "s ↦ h(x + s) + χ({‖s‖ ≤ Δ})"
fun_params(ψ::ShiftedNormL1BInf) = "x = $(ψ.x), Δ = $(ψ.Δ)"

function prox(ψ::ShiftedNormL1BInf{R, V1, V2}, q::AbstractVector{R}, σ::R) where {R <: Real, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  ProjB!(w) = begin 
    for i ∈ eachindex(w)
        w[i] = min(max(w[i], q[i] - ψ.λ * σ), q[i] + ψ.λ * σ)
    end
  end
  ProjΔ!(y) = begin 
    for ∈ eachindex(y)
        y[i] = min(max(y[i], -Δ), Δ)
    end
  end
  ψ.s .= -ψ.x
  ProjΔ!(ProjB!(ψ.s))
  return ψ.s
end