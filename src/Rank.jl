# Rank function

export Rank

"""
**``Rank`` lsc and subadditive **
    Rank(λ)
Returns the function
```math
f(x) = λ\\cdot rank(matrix(x))
```
for a nonnegative parameter `λ` and a vector `x`.
"""
struct Rank{R<: Real}
    lambda::R
    nrow::Int
    ncol::Int
    function Rank(lambda::R, nrow::Int, ncol::Int) where {R <: Real}
        if lambda < 0 || nrow <= 0 || ncol <= 0
            error("parameters λ, nrow and ncol must be nonnegative")
        end
        new{typeof(lambda)}(lambda, nrow, ncol)
    end
end

Rank(lambda::R, nrow::Int, ncol::Int) where {R} =  Rank{R}(lambda, nrow, ncol)

function (f::Rank)(x::AbstractVector{R}) where {R <: Real}
    return f.lambda * rank(reshape(x, f.nrow, f.ncol))
end

function prox!(y::AbstractVector{R}, f::Rank1{R}, x::AbstractVector{R}, gamma::R) where {R <: Real}
    A = reshape(x, f.nrow, f.ncol)
    F = svd(A)
    yvec = F.S
    c = sqrt(2 * f.lambda * gamma)
    for i ∈ eachindex(yvec)
      if abs(yvec[i]) <= c
        yvec[i] = 0
      end
    end
    return vec(reshape(F.U * Diagonal(yvec) * F.Vt, f.nrow * f.ncol, 1))
end

fun_name(f::Rank) = "Rank"
fun_dom(f::Rank) = "AbstractArray{Real}"
fun_expr(f::Rank{T}) where {T <: Real} = "x ↦ rank(matrix(x))"
fun_params(f::Rank{T}) where {T <: Real} = "λ = $(f.lambda)"