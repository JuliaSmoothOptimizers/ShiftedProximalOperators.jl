# Group ℓ₀ penalty, no cardinality window and no set constraint.
# prox is the separable hard threshold: keep Gᵢ when ‖x_{Gᵢ}‖² > 2γλᵢ.

export GroupNormL0

@doc raw"""
    GroupNormL0(λ, idx)
    GroupNormL0(λ, n::Integer)

Group \(\ell_0\) penalty
```math
    h(x) = \sum_{i=1}^m \lambda_i \|x_{G_i}\|_0,
```
with \(\|x_{G_i}\|_0 \in \{0,1\}\) and \(\lambda_i \ge 0\).
\(G_i\) is a group of components of \(x \in \mathbb{R}^n\) for \(i = 1,\ldots,m\),
with \(m = n\) the case where each group is a singleton.

`idx` is the list of groups (index vectors or ranges). The sets must be disjoint.
`prox!(y, f, x, γ)` writes the proximal point of \(h\) at `x` into `y`, and a
coordinate in no group is outside \(h\) so it stays equal to `x`.

`λ` is a scalar or a vector. `group_lambda` stores it as a float vector.
A scalar is used for every group. A vector has one weight per group.

`GroupNormL0(λ, n)` passes an integer instead of `idx`.
The groups are `[1:1, 2:2, ..., n:n]`. A vector `λ` then has length `n`.
"""
struct GroupNormL0{R <: Real, RR <: AbstractVector{R}, I}
  lambda::RR
  idx::I

  function GroupNormL0{R, RR, I}(lambda::RR, idx::I) where {R <: Real, RR <: AbstractVector{R}, I}
    m = length(idx)
    check_groups_disjoint(idx)
    length(lambda) == m || error("number of weights and groups must be the same")
    check_lambda_nonnegative(lambda)
    new{R, RR, I}(lambda, idx)
  end
end

function GroupNormL0(lambda::Union{Real, AbstractVector{<:Real}}, idx)
  lambda_v = group_lambda(lambda, length(idx))
  R = eltype(lambda_v)
  GroupNormL0{R, typeof(lambda_v), typeof(idx)}(lambda_v, idx)
end

# GroupNormL0(λ, n) passes an integer instead of `idx`.
# The groups are `[1:1, 2:2, ..., n:n]`. A vector `λ` then has length `n`.
function GroupNormL0(lambda::Union{Real, AbstractVector{<:Real}}, n::Integer)
  GroupNormL0(lambda, coordinate_groups(n))
end

function (f::GroupNormL0)(x::AbstractArray{R}) where {R <: Real}
  val, _ = group_penalty(f.lambda, f.idx, x)
  return val
end

function prox!(
  y::AbstractArray{R},
  f::GroupNormL0{R},
  x::AbstractArray{R},
  γ::R = one(R),
) where {R <: Real}
  # y receives prox_{γh}(x). Coordinates in no group stay equal to x.
  return group_hard_threshold!(y, x, f.idx, f.lambda, γ)
end

fun_name(f::GroupNormL0) = "group ℓ₀ penalty"
fun_dom(f::GroupNormL0) = "AbstractArray{<:Real}"
fun_expr(f::GroupNormL0) = "x ↦ Σᵢ λᵢ ‖x_{Gᵢ}‖₀"
fun_params(f::GroupNormL0) = "m = $(length(f.lambda))"
