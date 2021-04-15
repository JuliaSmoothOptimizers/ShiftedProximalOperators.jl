export ShiftedNormL0

mutable struct ShiftedNormL0{R <: Real, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} <: ShiftedProximableFunction
  h::NormL0{R}
  x::V1
  s::V2
  function ShiftedNormL0(h::NormL0{R}, x::AbstractVector{R})  where {R <: Real}
    s = similar(x)
    new{R, typeof(x), typeof(s)}(h, x, s)
  end
end

fun_name(ψ::ShiftedNormL0) = "shifted L0 pseudo-norm"

shifted(h::NormL0{R}, x::AbstractVector{R}) where {R <: Real} = ShiftedNormL0(h, x)

function prox(ψ::ShiftedNormL0{R, V1, V2}, q::AbstractVector{R}, σ::R) where {R <: Real, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} 
  c = sqrt(2 * ψ.λ * σ)
  s = similar(q)
  for i ∈ eachindex(q)
    xpq = ψ.x[i] + q[i]
    if abs(xpq) ≤ c
      ψ.s[i] = 0
    else
      ψ.s[i] = xpq
    end
  end
  ψ.s .-= ψ.x
  return ψ.s
end
