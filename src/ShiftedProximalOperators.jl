module ShiftedProximalOperators

using ProximalOperators
using Roots

export ShiftedProximableFunction
export prox, set_radius!, shift!, shifted

"Abstract type for shifted proximable functions."
abstract type ShiftedProximableFunction <: ProximableFunction end

include("shiftedNormL0.jl")
include("shiftedNormL0BInf.jl")

include("shiftedNormL1.jl")
include("shiftedNormL1B2.jl")
include("shiftedNormL1Binf.jl")
include("shiftedIndBallL0.jl")
include("shiftedIndBallL0BInf.jl")

(ψ::ShiftedProximableFunction)(y) = ψ.h(ψ.x0 + ψ.x + y)

"""
    shift!(ψ, x)

Update the shift of a shifted proximable function.
"""
function shift!(ψ::ShiftedProximableFunction, x::AbstractVector{R}) where {R <: Real}
  ψ.x .= x
  return ψ
end

"""
    set_radius!(ψ, Δ)

Set the trust-region radius of a shifted proximable function to Δ.
This method updates the indicator of the trust region that is part of ψ.
"""
function set_radius!(ψ::ShiftedProximableFunction, Δ::R) where {R <: Real}
  ψ.Δ = Δ
  return ψ
end

@inline function Base.getproperty(ψ::ShiftedProximableFunction, prop::Symbol)
  if prop === :λ
    return ψ.h.lambda
  elseif prop === :r
    return ψ.h.r
  else
    return getfield(ψ, prop)
  end
end

fun_name(ψ::ShiftedProximableFunction) = "undefined"
fun_expr(ψ::ShiftedProximableFunction) = "s ↦ h(x + s)"
fun_params(ψ::ShiftedProximableFunction) = "x0 = $(ψ.x0)\n" * " "^14 * "x = $(ψ.x)"

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

* ψ is a `ShiftedProximableFunction` representing a model of h(x + s) and
  possibly including the indicator of a trust region;
* q is the vector where the shifted proximal operator should be evaluated;
* σ is a positive regularization parameter.
"""
prox

"""
    shifted(h, x)
    shifted(h, x, Δ, χ)

Construct a shifted proximable function from a proximable function or from
another shifted proximable function.

If `h` is a `ProximableFunction`, including a `ShiftedProximableFunction`, the
form `shifted(h, x)` returns a `ShiftedProximableFunction` `ψ` such that `ψ(s)
== h(x + s)`. Subsequently, `prox` may be called on `ψ`. The first form applies
when `h` is a `ShiftedProximableFunction` and can be used to shift an
already-shifted proximable function.

The form `shifted(h, x, Δ, χ)` returns a `ShiftedProximableFunction` `ψ` such
that `ψ(s) == h(x + s) + Ind({‖s‖ ≤ Δ})`, where `Ind(.)` represents the
indicator of a set, in this case the indicator of a ball of radius `Δ`, in
which the norm is defined by `χ`.

### Arguments

* `h::ProximableFunction` (including a `ShiftedProximableFunction`)
* `x::AbstractVector`
* `Δ::Real`
* `χ::ProximableFunction`.

The currently supported combinations are:

* `h::IndBallL0` and `χ::Conjugate{IndBallL1}` (i.e., `χ` is the Inf-norm)
* `h::NormL0` and `χ::Conjugate{IndBallL1}` (i.e., `χ` is the Inf-norm)
* `h::NormL1` and `χ::Conjugate{IndBallL1}` (i.e., `χ` is the Inf-norm)
* `h::NormL1` and `χ::NormL2`.

If `h` is a shifted proximable function obtained from a previous call to
`shifted()`, only the form `shifted(h, x)` is supported. If applicable, the
resulting shifted proximable function is associated with the same `Δ` and `χ`
as `h`.

See the documentation of ProximalOperators.jl for more information.
"""
shifted

end # module
