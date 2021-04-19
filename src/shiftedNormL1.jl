export ShiftedNormL1

mutable struct ShiftedNormL1{R <: Real, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} <: ShiftedProximableFunction
  h::NormL1{R}
  x::V1
  s::V2
  function ShiftedNormL1(h::NormL1{R}, x::AbstractVector{R}) where {R <: Real}
    s = similar(x)
    new{R, typeof(x), typeof(s)}(h, x, s)
  end
end


shifted(h::NormL1{R}, x::AbstractVector{R}) where {R <: Real} = ShiftedNormL1(h, x)

fun_name(ψ::ShiftedNormL1) = "shifted L0 norm ball with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedNormL1) = "s ↦ h(x + s) + χ({‖s‖ ≤ Δ})"
fun_params(ψ::ShiftedNormL1) = "x = $(ψ.x), Δ = $(ψ.Δ)"

function prox(ψ::ShiftedNormL1{R, V1, V2}, q::AbstractVector{R}, σ::R) where {R <: Real, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  ProjB!(w) = begin 
    for i ∈ eachindex(w)
      w[i] = min(max(w[i], q[i] - ψ.λ * σ), q[i] + ψ.λ * σ)
    end
  end
  # find largest entries
  ψ.s .= -ψ.x

  ProjB!(ψ.s)
  return ψ.s 
end