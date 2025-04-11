# Group L2 norm (times a constant)

export GroupNormL2

@doc raw"""
    GroupNormL2(λ = 1, idx = [:])

Returns the group ``\ell_2``-norm operator
```math
f(x) =  \sum_i \lambda_i \|x_{[i]}\|_2^{1/2}
```
for groups ``x_{[i]}`` and nonnegative weights ``\lambda_i``.
The group ``\ell_2``-norm operator reduces to the ``\ell_2``-norm if only one group is defined (the default).
"""
struct GroupNormL2{R <: Real, RR <: AbstractVector{R}, I}
  lambda::RR
  idx::I

  function GroupNormL2{R, RR, I}(lambda::RR, idx::I) where {R <: Real, RR <: AbstractVector{R}, I}
    if any(lambda .< 0)
      error("weights λ must be nonnegative")
    elseif length(lambda) != length(idx)
      error("number of weights and groups must be the same")
    else
      new{R, RR, I}(lambda, idx)
    end
  end
end

GroupNormL2(lambda::AbstractVector{R} = [one(R)], idx::I = [:]) where {R <: Real, I} =
  GroupNormL2{R, typeof(lambda), I}(lambda, idx)

function (f::GroupNormL2)(x::AbstractArray{R}) where {R <: Real}
  sum_c = R(0)
  @views for (idx, λ) in zip(f.idx, f.lambda)
    sum_c += λ * norm(view(x, idx))
  end
  return sum_c
end

function prox!(
  y::AbstractArray{R},
  f::GroupNormL2{R, RR, I},
  x::AbstractArray{R},
  γ::R = R(1),
) where {R <: Real, RR <: AbstractVector{R}, I}
  ysum = R(0)
  for (idx, λ) ∈ zip(f.idx, f.lambda)
    yt = norm(x[idx])
    if yt == 0
      y[idx] .= 0
    else
      y[idx] .= max.(1 .- γ .* λ ./ yt, 0) .* x[idx]
      ysum += λ * yt
    end
  end
  return ysum
end

fun_name(f::GroupNormL2) = "Group L₂-norm"
fun_dom(f::GroupNormL2) = "AbstractArray{Float64}, AbstractArray{Complex}"
fun_expr(f::GroupNormL2) = "x ↦ Σᵢ λᵢ ‖xᵢ‖₂"
fun_params(f::GroupNormL2) = "λ = $(f.lambda), g = $(f.g)"
