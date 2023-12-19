# Group L2 norm (times a constant)

export GroupNormL0

@doc raw"""
    GroupNormL0(λ = 1, idx = [:])

Returns the group ``\ell_0``-norm operator
```math
f(x) =  \sum_i \lambda_i \| \|x_{[i]}\|_2 \|_0
```
for groups ``x_{[i]}`` and nonnegative weights ``\lambda_i``.
This assumes that the groups ``x_{[i]}`` are non-overlapping
"""
struct GroupNormL0{R <: Real, RR <: AbstractVector{R}, I}
  lambda::RR
  idx::I

  function GroupNormL0{R, RR, I}(lambda::RR, idx::I) where {R <: Real, RR <: AbstractVector{R}, I}
    if any(lambda .< 0)
      error("weights λ must be nonnegative")
    elseif length(lambda) != length(idx)
      error("number of weights and groups must be the same")
    else
      new{R, RR, I}(lambda, idx)
    end
  end
end

GroupNormL0(lambda::AbstractVector{R} = [one(R)], idx::I = [:]) where {R <: Real, I} =
  GroupNormL0{R, typeof(lambda), I}(lambda, idx)

function (f::GroupNormL0)(x::AbstractArray{R}) where {R <: Real}
  sum_c = R(0)
  for (idx, λ) ∈ zip(f.idx, f.lambda)
    y = norm(x[idx])
    if y>0
      sum_c += λ
    end
  end
  return sum_c
end

function prox!(
  y::AbstractArray{R},
  f::GroupNormL0{R, RR, I},
  x::AbstractArray{R},
  γ::R = R(1),
) where {R <: Real, RR <: AbstractVector{R}, I}
  ysum = R(0)
  for (idx, λ) ∈ zip(f.idx, f.lambda)
    yt = norm(x[idx])^2
    if yt !=0
      ysum += λ
    end
    if yt <= 2 * γ * λ
      y[idx] .= 0
    else
      y[idx] .= x[idx]
    end
  end
  return ysum
end

fun_name(f::GroupNormL0) = "Group L₀-norm"
fun_dom(f::GroupNormL0) = "AbstractArray{Float64}, AbstractArray{Complex}"
fun_expr(f::GroupNormL0) = "x ↦ Σᵢ λᵢ ‖ ‖xᵢ‖₂ ‖₀"
fun_params(f::GroupNormL0) = "λ = $(f.lambda), g = $(f.g)"
