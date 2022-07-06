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
reshape2(a, dims) = invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)

mutable struct Rank{R<: Real, S<:AbstractArray, T, Tr, M<:AbstractArray{T}} <: ProximableFunction
    lambda::R
    A::S
    F::PSVD{T,Tr,M}
    function Rank(lambda::R, A::S, F::PSVD{T,Tr,M}) where {R <: Real, S <: AbstractArray, T, Tr, M<:AbstractArray{T}}
        if lambda < 0 
            error("λ must be nonnegative")
        end
        new{typeof(lambda), typeof(A), T,Tr,M}(lambda, A, F)
    end
end


Rank(lambda::R, A::S, F::PSVD{T, Tr, M}) where {R,S,T,Tr, M<:AbstractArray{T}} =  Rank{R,S, eltype(A), real(eltype(A)) , M}(lambda, A, F)

function (f::Rank)(x::AbstractVector{R}) where {R <: Real}
    return f.lambda * rank(reshape2(x, (size(f.A,1), size(f.A,2))))
end


fun_name(f::Rank) = "Rank"
fun_dom(f::Rank) = "AbstractArray{Real}"
fun_expr(f::Rank{T}) where {T <: Real} = "x ↦ rank(matrix(x))"
fun_params(f::Rank{T}) where {T <: Real} = "λ = $(f.lambda)"


function prox!(y::AbstractVector{R}, f::Rank{R,S,T,Tr,M}, x::AbstractVector{R}, gamma::R) where {R <: Real, S <:AbstractArray, T, Tr, M<:AbstractArray{T}}
    #for i in eachindex(x)
    #    f.A[i] = x[i]
    #end
    f.A .= reshape2(x, (size(f.A,1),size(f.A,2))) # Allocation ici !!
    #psvd_dd!(f.F, reshape2(x, (size(f.A,1),size(f.A,2))), full=false)
    psvd_dd!(f.F, f.A, full=false) 
    c = sqrt(2 * f.lambda * gamma)
    for i ∈ eachindex(f.F.S)
        if f.F.S[i] <= c
            f.F.U[:,i] .= 0
        else for j in 1:size(f.A,1)
                f.F.U[j,i] = f.F.U[j,i] * f.F.S[i]
            end
        end
    end
    mul!(f.A, f.F.U, f.F.Vt)
    y .= reshape2(f.A, (size(y,1),1))
    return y
end

function proxqr!(y::AbstractVector{R}, f::Rank{R,S,T,Tr,M}, x::AbstractVector{R}, gamma::R) where {R <: Real, S <:AbstractArray, T, Tr, M<:AbstractArray{T}}
    #for i in eachindex(x)
    #    f.A[i] = x[i]
    #end
    f.A .= reshape2(x, (size(f.A,1),size(f.A,2))) # Allocation ici
    psvd_qr!(f.F, f.A, full=false) 
    c = sqrt(2 * f.lambda * gamma)
    for i ∈ eachindex(f.F.S)
        if f.F.S[i] <= c
            f.F.U[:,i] .= 0
        else for j in 1:size(f.A,1)
                f.F.U[j,i] = f.F.U[j,i] * f.F.S[i]
            end
        end
    end
    mul!(f.A, f.F.U, f.F.Vt)
    y .= reshape2(f.A, (size(y,1),1))
    return y
end



function prox_svd(y::AbstractVector{R}, f::Rank{R}, x::AbstractVector{R}, gamma::R) where {R <: Real}
    f.A .= reshape2(x, (size(f.A,1),size(f.A,2))) # Allocation ici en plus de SVD
    F = svd(f.A)
    c = sqrt(2 * f.lambda * gamma)
    for i ∈ eachindex(F.S)
        if F.S[i] <= c
            F.U[:,i] .= 0
        else for j in 1:size(f.A,1)
                F.U[j,i] = F.U[j,i] * F.S[i]
            end
        end
    end
    mul!(f.A, F.U, F.Vt)
    y .= reshape2(f.A, (size(y,1),1))
    return y
end
