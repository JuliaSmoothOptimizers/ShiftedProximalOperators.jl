# Group L2 norm (times a constant)

export GroupNormL2

"""
**``L_2`` Group - norm**
    GroupNormL2(λ = 1, g = 1, idx = [1, end])
Returns the function
```math
f(x) =  \\sum\\_{i} \\lambda\\_{i}||x\\_{[i]}||\\_2)^{1/2}
```
for groups `x\\_{[i]}` a nonnegative weights `λ\\_i`.
  Defaults to NormL2() in ProximalOperators if only 1 group is defined.
"""
struct GroupNormL2{
  R <: Real,
  RR <: AbstractVector{R},
  I
  } <: ProximableFunction
  lambda::RR
  idx::I

  function GroupNormL2{R, RR,I}(
    lambda::RR,
    idx::I
    ) where {R <: Real, RR <: AbstractVector{R}, I}
    if any(lambda .< 0)
      error("weights λ must be nonnegative")
    elseif length(lambda) != length(idx)
      error("number of weights and groups must be the same")
    else
      new{R, RR, I}(lambda, idx)
    end
  end
end

GroupNormL2(
  lambda::RR = [1.],
  idx::I = [Int[]],
) where {R <: Real, RR <: AbstractVector{R}, I} =
GroupNormL2{R, RR, I}(lambda, idx)

function (f::GroupNormL2)(x::AbstractArray{T}) where {T <: Real}
  sum_c = T(0)
  if length(f.idx) == 1
    return f.lambda * sqrt(sum(x.^2))
  else
    for i = 1:length(f.idx)
      sum_c += f.lambda[i]*sqrt(sum(x[f.idx[i]].^2))
    end
    return sum_c
  end
end

function prox!(
  y::AbstractArray{T},
  f::GroupNormL2{T, R, I},
  x::AbstractArray{T},
  γ::T = T(1),
) where {T <: Real, R <: AbstractVector{T}, I}

  ysum = T(0)
  yt = T(0)
  if length(f.idx) == 1
    ysum = f.lambda*norm(x)
    y .= max(1 .- γ * f.lambda / ysum, 0) .* x
  else
    for i = 1:length(f.idx)
      yt = norm(x[f.idx[i]])
      ysum += f.lambda[i]*yt
      y[f.idx[i]] .= max(1 .- γ*f.lambda[i]/yt, 0) .* x[f.idx[i]]
    end
  end
  return ysum
end

fun_name(f::GroupNormL2) = "Group L₂-norm"
fun_dom(f::GroupNormL2) = "AbstractArray{Float64}, AbstractArray{Complex}"
fun_expr(f::GroupNormL2) = "x ↦ Σᵢ λᵢ ‖xᵢ‖₂"
fun_params(f::GroupNormL2) = "λ = $(f.lambda), g = $(f.g)"

