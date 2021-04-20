
export ShiftedIndBallL0BInf

mutable struct ShiftedIndBallL0BInf{R <: Real, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} <: ShiftedProximableFunction
  h::IndBallL0{R}
  x::V1
  s::V2
  Δ::R
  χ::Conjugate{IndBallL1{R}}
  function ShiftedIndBallL0BInf(h::IndBallL0{R}, x::AbstractVector{R}, Δ::R, χ::Conjugate{IndBallL1{R}}) where {R <: Real}
    s = similar(x)
    new{R, typeof(x), typeof(s)}(h, x, s, Δ)
  end
end


shifted(h::IndBallL0{R}, x::AbstractVector{R}, Δ::R, χ::Conjugate{IndBallL1{R}}) where {R <: Real} = ShiftedIndBallL0BInf(h, x, Δ, χ)

fun_name(ψ::ShiftedIndBallL0BInf) = "shifted L0 norm ball with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedIndBallL0BInf) = "s ↦ h(x + s) + χ({‖s‖∞ ≤ Δ})"
fun_params(ψ::ShiftedIndBallL0BInf) = "x = $(ψ.x), Δ = $(ψ.Δ)"

function prox(ψ::ShiftedIndBallL0BInf{R, V1, V2}, q::AbstractVector{R}, σ::R) where {R <: Real, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  ProjB!(w) = begin 
    for i ∈ eachindex(w)
      w[i] = min(max(w[i], ψ.x[i] - ψ.Δ), ψ.x[i] + ψ.Δ)
    end
  end
  
  q .+= ψ.x
  # find largest entries
  ψ.s .= sortperm(q, rev=true, by = abs) #stock with ψ.s as placeholder
  q[ψ.s[ψ.h.r + 1:end]] .= 0 # set smallest to zero 
  ProjB!(q)# put all entries in projection?
  ψ.s .= q - ψ.x 
  return ψ.s 
end