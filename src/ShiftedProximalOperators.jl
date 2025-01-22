module ShiftedProximalOperators

using LinearAlgebra
using QRMumps
using SparseMatricesCOO

using libblastrampoline_jll
using OpenBLAS32_jll
using ProximalOperators
using Roots
using SparseArrays

function __init__()
  # Ensure LBT points to a valid BLAS for psvd()
  if VERSION ≥ v"1.7"
    config = LinearAlgebra.BLAS.lbt_get_config()
    if !any(lib -> lib.interface == :lp64, config.loaded_libs)
      LinearAlgebra.BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
    end
  end
end

export ShiftedProximableFunction
export prox, prox!, iprox, iprox!, set_radius!, shift!, shifted, set_bounds!

# import methods we override
import ProximalOperators.prox, ProximalOperators.prox!

"Abstract type for shifted proximable functions."
abstract type ShiftedProximableFunction end
abstract type CompositeProximableFunction end

abstract type AbstractCompositeNorm <: CompositeProximableFunction end
abstract type ShiftedCompositeProximableFunction <: ShiftedProximableFunction end

include("utils.jl")
include("psvd.jl")

include("compositeNormL2.jl")
include("rootNormLhalf.jl")
include("groupNormL2.jl")
include("Rank.jl")
include("cappedl1.jl")
include("Nuclearnorm.jl")

include("shiftedCompositeNormL2.jl")
include("shiftedNormL0.jl")
include("shiftedNormL0Box.jl")
include("shiftedRootNormLhalf.jl")
include("shiftedNormL1.jl")
include("shiftedGroupNormL2.jl")

include("shiftedNormL1B2.jl")
include("shiftedNormL1Box.jl")
include("shiftedIndBallL0.jl")
include("shiftedIndBallL0BInf.jl")
include("shiftedRootNormLhalfBox.jl")
include("shiftedGroupNormL2Binf.jl")
include("shiftedRank.jl")
include("shiftedCappedl1.jl")
include("shiftedNuclearnorm.jl")

function (ψ::ShiftedProximableFunction)(y)
  @. ψ.xsy = ψ.xk + ψ.sj + y
  return ψ.h(ψ.xsy)
end

function (ψ::ShiftedCompositeProximableFunction)(y)
  return ψ.h(ψ.b + ψ.A * y)
end

function (ψ::CompositeProximableFunction)(y)
  z = similar(ψ.b)
  ψ.c!(z, y)
  ψ.h(z)
end

"""
    shift!(ψ, x)

Update the shift of a shifted proximable function.
"""
function shift!(ψ::ShiftedProximableFunction, shift::AbstractVector{R}) where {R <: Real}
  if ψ.shifted_twice
    ψ.sj .= shift
  else
    ψ.xk .= shift
  end
  return ψ
end

function shift!(ψ::ShiftedCompositeProximableFunction, shift::AbstractVector{R}) where {R <: Real}
  ψ.c!(ψ.b,shift)
  ψ.J!(ψ.A,shift)
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
set_radius!(ψ::ShiftedNormL0Box, Δ::R) where {R <: Real} = set_bounds!(ψ, -Δ, Δ)
set_radius!(ψ::ShiftedNormL1Box, Δ::R) where {R <: Real} = set_bounds!(ψ, -Δ, Δ)
set_radius!(ψ::ShiftedRootNormLhalfBox, Δ::R) where {R <: Real} = set_bounds!(ψ, -Δ, Δ)

"""
    set_bounds!(ψ, l, u)

Set the lower and upper bound of the box to l and u, respectively.
l and u can be scalars or vectors.
"""
function set_bounds!(ψ::ShiftedProximableFunction, l, u)
  isa(l, Real) ? ψ.l = l : (isa(ψ.l, Real) ? ψ.l = l : ψ.l .= l)
  isa(u, Real) ? ψ.u = u : (isa(ψ.u, Real) ? ψ.u = u : ψ.u .= u)
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
fun_expr(ψ::ShiftedProximableFunction) = "t ↦ h(xk + sj + t)"
fun_params(ψ::ShiftedProximableFunction) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)"

function Base.show(io::IO, ψ::ShiftedProximableFunction)
  println(io, "description : ", fun_name(ψ))
  println(io, "expression  : ", fun_expr(ψ))
  println(io, "parameters  : ", fun_params(ψ))
  println(io, "\nunderlying proximable function h:")
  show(io, ψ.h)
end

"""
    prox!(y, ψ, q, σ)

Evaluate the proximal operator of a shifted regularizer, i.e, return
a solution s of

    minimize{s}  ½ σ⁻¹ ‖s - q‖₂² + ψ(s),

where

* ψ is a `ShiftedProximableFunction` representing a model of h(x + s) and
  possibly including the indicator of a trust region;
* q is the vector where the shifted proximal operator should be evaluated;
* σ is a positive regularization parameter.

The solution is stored in the input vector `y` an `y` is returned.
"""
prox!

"""
    iprox!(y, ψ, g, d)

Evaluate the indefinite proximal operator of a separable box shifted regularizer, i.e, return
a solution y of

    minimize{y}  ½ yᵀDy + gᵀy + ψ(y)

where

* ψ is a `ShiftedProximableFunction` representing a model of the sum of a separable function h(x + s) and
  the indicator of a trust region;
* g is a vector;
* `D = Diagonal(d)` where d is a vector.

The solution is stored in the input vector `y` an `y` is returned.
"""
iprox!

"""
    iprox(ψ, g, d)

See the documentation of `iprox!`.
In this form, the solution is stored in ψ's internal storage and a reference
is returned.
"""
iprox(ψ::ShiftedProximableFunction, g::AbstractVector, d::AbstractVector) = iprox!(ψ.sol, ψ, g, d)

"""
    prox(ψ, q, σ)

See the documentation of `prox!`.
In this form, the solution is stored in ψ's internal storage and a reference
is returned.
"""
prox(ψ::ShiftedProximableFunction, q::V, σ::R) where {R <: Real, V <: AbstractVector{R}} =
  prox!(ψ.sol, ψ, q, σ)

"""
    prox_zero(q, l, u)

Return the solution of

    min ½ σ⁻¹ (y - q)² subject to l ≤ y ≤ u

for any σ > 0. This problem occurs when computing the prox with respect to a
separable nonsmooth term along a variable that is not part of those to which
the nonsmooth term is applied.
"""
@inline prox_zero(q::R, l::R, u::R) where {R <: Real} = min(max(q, l), u)

"""
    iprox_zero(d, g, l, u)

Return the solution of

    min ½ d y² + g y subject to l ≤ y ≤ u

for any d, g, assuming that 0 ∈ [l, u] if d < eps() and g = 0.
This problem occurs when computing the iprox with respect to a
separable nonsmooth term along a variable that is not part of those to which
the nonsmooth term is applied.
"""
function iprox_zero(d::R, g::R, l::R, u::R) where {R <: Real}
  if d > eps(R)
    argmin_quad = -g / d
    y = min(max(argmin_quad, l), u)
  elseif d < -eps(R)
    d_2 = d / 2
    val_l = d_2 * l^2 + g * l
    val_u = d_2 * u^2 + g * u
    y = (val_l < val_u) ? l : u
  else # abs(d) ≤ eps(R)
    if g > zero(R)
      y = l
    elseif g < zero(R)
      y = u
    else # g == 0
      y = zero(R)
    end
  end
  return y
end

"""
    shifted(h, x)
    shifted(h, x, Δ, ρ)

Construct a shifted proximable function from a proximable function or from
another shifted proximable function.

If `h` is a `ProximableFunction`, including a `ShiftedProximableFunction`, the
form `shifted(h, x)` returns a `ShiftedProximableFunction` `ψ` such that `ψ(s)
== h(x + s)`. Subsequently, `prox` may be called on `ψ`. The first form applies
when `h` is a `ShiftedProximableFunction` and can be used to shift an
already-shifted proximable function.

The form `shifted(h, x, Δ, ρ)` returns a `ShiftedProximableFunction` `ψ` such
that `ψ(s) == h(x + s) + Ind({‖s‖ ≤ Δ})`, where `Ind(.)` represents the
indicator of a set, in this case the indicator of a ball of radius `Δ`, in
which the norm is defined by `ρ`.

### Arguments

* `h::ProximableFunction` (including a `ShiftedProximableFunction`)
* `x::AbstractVector`
* `Δ::Real`
* `ρ::ProximableFunction`.

Only certain combinations of `h` and `ρ` are supported; those for which the
analytical form of the proximal operator is known.

If `h` is a shifted proximable function obtained from a previous call to
`shifted()`, only the form `shifted(h, x)` is supported. If applicable, the
resulting shifted proximable function is associated with the same `Δ` and `ρ`
as `h`.

See the documentation of ProximalOperators.jl for more information.
"""
shifted

end # module
