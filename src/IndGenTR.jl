# Indicator of a generalized trust region.

export IndGenTR

"""
Indicator function of a trust region defined by two n-dimensionnal vectors l and u.
Check component wise if the elements are between the components of l and u.
"""

struct IndGenTR
    l::AbstractVector{Real}
    u::AbstractVector{Real}
    function IndGenTR(l::AbstractVector{Real},u::AbstractVector{Real})
        if sum(l .> u) != 0
            error("Out of order.")
        else
            new(l,u)
        end
    end
end

# input : x, AbstractVector{Real}, vector of real elements
# output : ind, AbstractVector{Real}, element wise indicator

function (f::IndGenTR)(x::AbstractVector{Real})
    ind = zero(x)
    ind[.!(f.l .<= x .<= f.u)] .= Inf
    return ind
end

# Example :
# include("IndGenTR.jl")
# l = Vector{Real}([0.0, 0.0]) ; u = Vector{Real}([1.0, 1.0])
# f = IndGenTR(l,u)
# x = Vector{Real}([0.5,10])
# f(x)