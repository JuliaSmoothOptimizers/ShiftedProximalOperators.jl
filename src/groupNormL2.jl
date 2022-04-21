# Group L2 norm (times a constant)

export GroupNormL2

"""
**``L_2`` Group - norm**
    GroupNormL2(λ=1)
Returns the function
```math
f(x) = λ\\cdot \\sum\\_{i} ||x\\_{[i]}||\\_2)^{1/2}
```
for a nonnegative parameter `λ`.
"""
struct GroupNormL2{R <: Real, I <: Integer, V0 <: AbstractVector{I}} <: ProximableFunction
  lambda::R
  g::I
  idx::V0
  function GroupNormL2{R}(lambda::R, g::I, idx::AbstractVector{I}) where {R <: Real, I <: Integer}
    if lambda < 0
      error("parameter λ must be nonnegative")
    elseif Groups < 1
      error("Must have more than one group")
    elseif mod(size(idx,2),2)~=0
      error("Must provide indices as 2 x i")
    elseif Groups ~= size(idx,2)
      error("Number of groups must be the same as length of indices provided")
    else
      new{R,I,typeof(idx)}(lambda, g, idx)
    end
  end
end

GroupNormL2(lambda::R = 1, g::I = 1, idx::AbstractVector{I} = [1,end]) where {R <: Real, I <: IntegerV0 <: AbstractVector{I}} = GroupNormL2(lambda, g, idx)

function (f::GroupNormL2)(x::AbstractArray{T}) where {T <: Real}
  sum_c = 0
  for i = 1:f.g
    sum_c += sqrt(sum(x[idx[1,i]:idx[2,i]].^2))
  end
  return f.lambda * T(sum)
end

function prox!(
  y::AbstractArray{T},
  f::RootNormLhalf,
  x::AbstractArray{T},
  gamma::Real = 1,
) where {T <: Real}
 
  ysum = 0
  for i = 1:f.g
    ysum += sqrt(sum(x[idx[1,i]:idx[2,i]].^2))
    y[idx[1,i]:idx[2,i]] .= max(1 - γ/norm(x[idx[1,i]:idx[2,i]]), 0) .* x[idx[1,i]:idx[2,i]]
  end

  
  return f.lambda * ysum
end

fun_name(f::GroupNormL2) = "Group L^2-norm"
fun_dom(f::GroupNormL2) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::GroupNormL2{T}) where {T <: Real} = "x ↦ ½ λ \sum_i ‖x_[i]‖_(2)^(½)"
fun_params(f::GroupNormL2{T}) where {T <: Real} = "λ = $(f.lambda)"

