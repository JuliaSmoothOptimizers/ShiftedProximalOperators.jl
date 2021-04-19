export ShiftedIndBallL0

mutable struct ShiftedIndBallL0{R <: Real, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} <: ShiftedProximableFunction
  h::IndBallL0{R}
  x::V1
  s::V2
  function ShiftedIndBallL0(h::IndBallL0{R}, x::AbstractVector{R}) where {R <: Real}
    s = similar(x)
    new{R, typeof(x), typeof(s)}(h, x, s)
  end
end


shifted(h::IndBallL0{R}, x::AbstractVector{R}) where {R <: Real} = ShiftedIndBallL0(h, x)

fun_name(ψ::ShiftedIndBallL0) = "shifted L0 norm ball with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedIndBallL0) = "s ↦ h(x + s) + χ({‖s‖ ≤ Δ})"
fun_params(ψ::ShiftedIndBallL0) = "x = $(ψ.x), Δ = $(ψ.Δ)"

function prox(ψ::ShiftedIndBallL0{R, V1, V2}, q::AbstractVector{R}, σ::R) where {R <: Real, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  q .+= ψ.x
  # find largest entries
  ψ.s .= sortperm(q, rev=true, by = abs) #stock with ψ.s as placeholder
  q[ψ.s[ψ.h.r + 1:end]] .= 0 # set smallest to zero 
  ProjB!(q)# put all entries in projection?
  ψ.s .= q - ψ.x 
  return ψ.s 
end