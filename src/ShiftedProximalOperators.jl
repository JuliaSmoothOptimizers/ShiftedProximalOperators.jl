module ShiftedProximalOperators

using ProximalOperators, Roots, LinearAlgebra

export ShiftedProximableFunction
export prox, set_radius!, shift!, shifted

abstract type ShiftedProximableFunction <: ProximableFunction end

include("shiftedNormL0.jl")
# include("shiftedNormL0BInf.jl")

include("shiftedNormL1.jl")
include("shiftedNormL1Binf.jl")
include("shiftedNormL1B2.jl")

include("shiftedIndBallL0.jl")
include("shiftedIndBallL0BInf.jl")

(ψ::ShiftedProximableFunction)(y) = ψ.h(ψ.x + y)

function shift!(ψ::ShiftedProximableFunction, x::AbstractVector{R}) where {R <: Real}
	ψ.x .= x
	return ψ
end

function set_radius!(ψ::ShiftedProximableFunction, Δ::R) where {R <: Real}
  ψ.Δ = Δ
  return ψ
end

@inline function Base.getproperty(ψ::ShiftedProximableFunction, prop::Symbol)
  return prop == :λ ? ψ.h.lambda : getfield(ψ, prop)
end

fun_name(ψ::ShiftedProximableFunction) = "undefined"
fun_expr(ψ::ShiftedProximableFunction) = "s ↦ h(x + s)"
fun_params(ψ::ShiftedProximableFunction) = "x = $(ψ.x)"

function Base.show(io::IO, ψ::ShiftedProximableFunction)
  println(io, "description : ", fun_name(ψ))
  println(io, "expression  : ", fun_expr(ψ))
  println(io, "parameters  : ", fun_params(ψ))
  println(io, "\nunderlying proximable function h:")
  show(io, ψ.h)
end

"""
    prox(ψ, q, σ)

Evaluate the proximal operator of a shifted regularizer, i.e, return
a solution s of

    minimize{s}  ½σ ‖s - q‖₂² + ψ(s),

where

* ψ is a `ShiftedProximableFunction` representing a model of h(x + s);
* q is the vector where the shifted proximal operator should be evaluated;
* σ is a positive regularization parameter.
"""
prox

end # module
