export ShiftedNormL1

mutable struct ShiftedNormL1{R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} <: ShiftedProximableFunction
  h::NormL1{R}
  x0::V0 
  x::V1
  s::V2
  function ShiftedNormL1(h::NormL1{R}, x0::AbstractVector{R}, x::AbstractVector{R}) where {R <: Real}
    s = similar(x)
    new{R, typeof(x0), typeof(x), typeof(s)}(h, x0, x, s)
  end
end

shifted(h::NormL1{R}, x::AbstractVector{R}) where {R <: Real} = ShiftedNormL1(h, zero(x), x)
shifted(ψ::ShiftedNormL1{R, V0, V1, V2}, x::AbstractVector{R}) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} = ShiftedNormL1(ψ.h, ψ.x, x)

fun_name(ψ::ShiftedNormL1) = "shifted L1 norm"
fun_expr(ψ::ShiftedNormL1) = "s ↦ h(x + s)"
fun_params(ψ::ShiftedNormL1) = "x = $(ψ.x), Δ = $(ψ.Δ)"

function prox(ψ::ShiftedNormL1{R, V0, V1, V2}, q::AbstractVector{R}, σ::R) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  ProjB!(w) = begin 
    for i ∈ eachindex(w)
      w[i] = min(max(w[i], q[i] - ψ.λ * σ), q[i] + ψ.λ * σ)
    end
  end
  # find largest entries
  ψ.s .-= ψ.x .+ ψ.x0

  ProjB!(ψ.s)
  return ψ.s 
end
