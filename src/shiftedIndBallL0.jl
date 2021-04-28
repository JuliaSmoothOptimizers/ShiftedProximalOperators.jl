export ShiftedIndBallL0

mutable struct ShiftedIndBallL0{I <: Integer, R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} <: ShiftedProximableFunction
  h::IndBallL0{I}
  x0::V0
  x::V1
  s::V2
  p::Vector{Int}
  function ShiftedIndBallL0(h::IndBallL0{I}, x0::AbstractVector{R}, x::AbstractVector{R}) where {I <: Integer, R <: Real}
    s = similar(x)
    new{I, R, typeof(x0), typeof(x), typeof(s)}(h, x0, x, s, Vector{Int}(undef, length(x)))
  end
end

shifted(h::IndBallL0{I}, x::AbstractVector{R}) where {I <: Integer, R <: Real} = ShiftedIndBallL0(h, zero(x), x)
shifted(ψ::ShiftedIndBallL0{I, R, V0, V1, V2}, x::AbstractVector{R}) where {I <: Integer, R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} = ShiftedIndBallL0(ψ.h, ψ.x, x)

fun_name(ψ::ShiftedIndBallL0) = "shifted L0 norm ball indicator"

function prox(ψ::ShiftedIndBallL0{I, R, V0, V1, V2}, q::AbstractVector{R}, σ::R) where {I <: Integer, R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  q .+= (ψ.x + ψ.x0)
  # find largest entries
  sortperm!(ψ.p, q, rev=true, by = abs) #stock with ψ.s as placeholder
  q[ψ.p[ψ.h.r + 1:end]] .= 0 # set smallest to zero 
  ψ.s .= q - (ψ.x + ψ.x0) 
  return ψ.s 
end
