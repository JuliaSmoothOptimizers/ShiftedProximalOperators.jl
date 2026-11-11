# Group ℓ₀ penalty plus a cardinality window.
#
# For l = 0 the lower bound never forces a block on, so Dᵢ = ℝ^{nᵢ}. The
# candidate is the block of x. The exclusion ball is not used.
#
# For l > 0 the model is not the whole space. For the prox to be attained, an
# active block has to stay away from the origin: otherwise a lower bound l, or
# a start at 0, asks for a nonzero block whose cost can be decreased by
# shrinking it, and the infimum is not attained. Each group is therefore constrained to
#   Dᵢ = { z ∈ ℝ^{nᵢ} : ‖z‖ ≥ δ } = ℝ^{nᵢ} \ { z : ‖z‖ < δ },
# the complement of the open ball of radius δ. The candidate written for an
# active group is the Euclidean projection onto Dᵢ.
# 
# Outside the ball that projection is the block of x. Inside, it lies on the
# sphere of radius δ, in the direction of x. At a zero block every point of the
# sphere is nearest; the first coordinate in the group's index order is set to
# +δ and the others to 0. A scalar block then equals ±δ.
#
# δ defaults to eps of the weight type. A block placed on that sphere is nonzero,
# and the quadratic cost of the jump is ½δ², negligible next to an ordinary penalty.
# That default warns, and names every selected group that landed on the sphere.
# A caller-supplied radius never warns.

export GroupRankL0

@doc raw"""
    GroupRankL0(λ, idx, l, s; delta = sqrt(eps))
    GroupRankL0(λ, n::Integer, l, s; delta = sqrt(eps))

Group \(\ell_0\) penalty with a cardinality window
```math
    h(x) = \sum_{i=1}^m \lambda_i \|x_{G_i}\|_0 + \chi_{B \cap C_{l,s}}(x),
```
with \(\|x_{G_i}\|_0 \in \{0,1\}\) and \(\lambda_i \ge 0\).
\(G_i\) is a group of components of \(x \in \mathbb{R}^n\) for \(i = 1,\ldots,m\),
with \(m = n\) the case where each group is a singleton.
\(B = \prod_i (D_i \cup \{0\})\). \(C_{l,s}\) is the set of vectors with at least \(l\)
and at most \(s\) active groups, where group \(i\) is active when \(x_{G_i} \neq 0\).
For `l > 0`, \(D_i\) is the complement of the open ball of radius \(\delta\).
\(\delta\) defaults to the square root of eps of the weight type.
That default warns, and names every selected group that landed on the sphere.
A caller-supplied radius never warns. For `l = 0`, \(D_i = \mathbb{R}^{n_i}\).

`idx` is the list of groups (index vectors or ranges). The sets must be disjoint.
`prox!(y, f, x, γ)` writes the proximal point of \(h\) at `x` into `y`, and a
coordinate in no group is outside \(h\) so it stays equal to `x`.

`λ` is a scalar or a vector. `group_lambda` stores it as a float vector.
A scalar is used for every group. A vector has one weight per group.

`GroupRankL0(λ, n, l, s; delta = sqrt(eps))` passes an integer instead of `idx`.
`l`, `s`, and `delta` are unchanged, and the groups are `[1:1, 2:2, ..., n:n]`.
A vector `λ` then has length `n`.

Shifted gains are \(\rho_i = \omega_i - 2\gamma\lambda_i\), with
\(\omega_i = \|x_{G_i}\|^2 - \|x_{G_i} - P_{D_i}(x_{G_i})\|^2\),
which are sorted and then ranked against zero by the filter in `groupL0support.jl`.

`l = 0`, `s = m` keeps a group when \(\rho_i > 0\) and does not sort.
"""
struct GroupRankL0{R <: Real, RR <: AbstractVector{R}, I}
  lambda::RR
  idx::I
  l::Int
  s::Int
  delta::R
  default_delta::Bool
  rho::Vector{R}
  perm::Vector{Int}
  snapped::Vector{Bool}
  proj::Vector{R}
  proj_at::Vector{Int}

  function GroupRankL0{R, RR, I}(
    lambda::RR,
    idx::I,
    l::Int,
    s::Int,
    delta::R,
    default_delta::Bool,
  ) where {R <: Real, RR <: AbstractVector{R}, I}
    m = length(idx)
    check_groups_disjoint(idx)
    length(lambda) == m || error("number of weights and groups must be the same")
    check_lambda_nonnegative(lambda)
    isfinite(delta) && delta > 0 || error("exclusion radius δ must be positive and finite")
    at, nproj = proj_layout(idx)
    new{R, RR, I}(
      lambda,
      idx,
      l,
      s,
      delta,
      default_delta,
      Vector{R}(undef, m),
      collect(1:m),
      fill(false, m),
      Vector{R}(undef, nproj),
      at,
    )
  end
end

function GroupRankL0(
  lambda::Union{Real, AbstractVector{<:Real}},
  idx,
  l::Integer,
  s::Integer;
  delta::Union{Nothing, Real} = nothing,
)
  lambda_v = group_lambda(lambda, length(idx))
  R = eltype(lambda_v)
  ll, ss = group_window(l, s, length(idx))
  # Omitted delta is sqrt(eps(R)). 
  # A passed value, even sqrt(eps(R)) itself, is the caller's scale.
  default_delta = delta === nothing
  δ = default_delta ? sqrt(eps(R)) : R(delta)
  GroupRankL0{R, typeof(lambda_v), typeof(idx)}(lambda_v, idx, ll, ss, δ, default_delta)
end

# GroupRankL0(λ, n, l, s; delta = sqrt(eps)) passes an integer instead of `idx`.
# The groups are `[1:1, 2:2, ..., n:n]`. A vector `λ` then has length `n`.
function GroupRankL0(
  lambda::Union{Real, AbstractVector{<:Real}},
  n::Integer,
  l::Integer,
  s::Integer;
  delta::Union{Nothing, Real} = nothing,
)
  GroupRankL0(lambda, coordinate_groups(n), l, s; delta)
end

function block_inside_ball(x, g, δ)
  xg = view(x, g)
  nrm2 = zero(real(eltype(x)))
  active = false
  for t in eachindex(xg)
    active |= xg[t] != 0
    nrm2 += xg[t] * xg[t]
  end
  return active && sqrt(nrm2) < δ
end

function (f::GroupRankL0)(x::AbstractArray{R}) where {R <: Real}
  if f.l > 0
    for g in f.idx
      if block_inside_ball(x, g, f.delta)
        return R(Inf)
      end
    end
  end
  val, nact = group_penalty(f.lambda, f.idx, x)
  if nact < f.l || nact > f.s
    return typeof(val)(Inf)
  end
  return val
end

# Selected groups that the default ball moved onto the sphere.
function warn_default_ball!(f::GroupRankL0, k::Int)
  f.default_delta || return
  n = 0
  @inbounds for t in 1:k
    f.snapped[f.perm[t]] && (n += 1)
  end
  n == 0 && return
  ids = Vector{Int}(undef, n)
  p = 1
  @inbounds for t in 1:k
    i = f.perm[t]
    if f.snapped[i]
      ids[p] = i
      p += 1
    end
  end
  sort!(ids)
  @warn "default exclusion ball placed groups $(ids) at radius ±$(f.delta)"
  return
end

function prox!(
  y::AbstractArray{R},
  f::GroupRankL0{R},
  x::AbstractArray{R},
  γ::R = one(R),
) where {R <: Real}
  # y receives prox_{γh}(x). Coordinates in no group stay equal to x.
  # l = 0 never forces a block on, so Dᵢ = ℝ^{nᵢ} and the candidate is x.
  if f.l == 0
    fill_copy_gains!(f.rho, x, f.idx, f.lambda, γ)
    k = group_l0_support!(f.perm, f.rho, f.l, f.s)
    return write_group_support!(y, x, f.idx, f.lambda, f.perm, k, x, nothing)
  end
  fill_ball_gains!(f.rho, f.proj, f.proj_at, f.snapped, x, f.idx, f.lambda, γ, f.delta)
  k = group_l0_support!(f.perm, f.rho, f.l, f.s)
  warn_default_ball!(f, k)
  return write_group_support!(y, x, f.idx, f.lambda, f.perm, k, f.proj, f.proj_at)
end

fun_name(f::GroupRankL0) = "group ℓ₀ penalty with cardinality"
fun_dom(f::GroupRankL0) = "AbstractArray{<:Real}"
fun_expr(f::GroupRankL0) = "x ↦ Σᵢ λᵢ ‖x_{Gᵢ}‖₀ + χ_{B ∩ C_{l,s}}(x)"
fun_params(f::GroupRankL0) =
  "m = $(length(f.lambda)), l = $(f.l), s = $(f.s), δ = $(f.delta)"
