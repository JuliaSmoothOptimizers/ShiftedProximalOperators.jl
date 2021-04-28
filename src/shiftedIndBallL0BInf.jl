export ShiftedIndBallL0BInf

mutable struct ShiftedIndBallL0BInf{I <: Integer, R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} <: ShiftedProximableFunction
  h::IndBallL0{I} #this only takes integers 
  x0::V0
  x::V1
  s::V2
  p::Vector{Int}
  Δ::R
  χ::Conjugate{IndBallL1{R}}
  function ShiftedIndBallL0BInf(h::IndBallL0{I}, x0::AbstractArray{R}, x::AbstractVector{R}, Δ::R, χ::Conjugate{IndBallL1{R}}) where {I <: Integer, R <: Real}
    s = similar(x)
    new{I, R, typeof(x0), typeof(x), typeof(s)}(h, x0, x, s, Vector{Int}(undef, length(x)), Δ, χ)
  end
end


shifted(h::IndBallL0{I}, x::AbstractVector{R}, Δ::R, χ::Conjugate{IndBallL1{R}}) where {I <: Integer, R <: Real} = ShiftedIndBallL0BInf(h,zero(x), x, Δ, χ)
shifted(ψ::ShiftedIndBallL0BInf{I, R, V0, V1, V2}, x::AbstractVector{R}) where {I <: Integer, R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} = ShiftedIndBallL0BInf(ψ.h, ψ.x, x, Δ, χ)

fun_name(ψ::ShiftedIndBallL0BInf) = "shifted L0 norm ball with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedIndBallL0BInf) = "s ↦ h(x + s) + χ({‖s‖∞ ≤ Δ})"
fun_params(ψ::ShiftedIndBallL0BInf) = "x0 = $(ψ.x0)\n" * " "^14 * "x = $(ψ.x), Δ = $(ψ.Δ)"

function prox(ψ::ShiftedIndBallL0BInf{I, R, V0, V1, V2}, q::AbstractVector{R}, σ::R) where {I<: Integer, R <: Real,  V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  ProjB!(w) = begin 
    for i ∈ eachindex(w)
      w[i] = min(max(w[i], (ψ.x[i] + ψ.x0[i]) - ψ.Δ), (ψ.x[i] + ψ.x0[i]) + ψ.Δ)
    end
  end
  ψ.s .= q
  ψ.s .+= (ψ.x + ψ.x0)
  # find largest entries
  sortperm!(ψ.p, ψ.s, rev = true, by = abs) #stock with ψ.s as placeholder
  ψ.s[ψ.p[ψ.h.r + 1:end]] .= 0 # set smallest to zero
  ProjB!(ψ.s)
  ψ.s .-= (ψ.x + ψ.x0) 
  return ψ.s 
end
