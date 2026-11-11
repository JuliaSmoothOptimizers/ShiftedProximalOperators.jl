# Group ℓ₀ penalty, cardinality window, and a product of sets.
# B = ∏ᵢ (Dᵢ ∪ {0}). For l > 0, 0 ∉ Dᵢ. For l = 0, project! may return 0.
# The projection onto Dᵢ is computed once per group, stored,
# and reused for every group that the filter keeps.

export GroupRankProjL0

@doc raw"""
    GroupRankProjL0(λ, idx, l, s, project!)
    GroupRankProjL0(λ, n::Integer, l, s, project!)

Group \(\ell_0\) penalty with a cardinality window and a product of sets
```math
    h(x) = \sum_{i=1}^m \lambda_i \|x_{G_i}\|_0 + \chi_{B \cap C_{l,s}}(x),
```
with \(\|x_{G_i}\|_0 \in \{0,1\}\) and \(\lambda_i \ge 0\).
\(G_i\) is a group of components of \(x \in \mathbb{R}^n\) for \(i = 1,\ldots,m\),
with \(m = n\) the case where each group is a singleton.
\(B = \prod_i (D_i \cup \{0\})\). \(C_{l,s}\) is the set of vectors with at least \(l\)
and at most \(s\) active groups, where group \(i\) is active when \(x_{G_i} \neq 0\).
For `l > 0`, \(0 \notin D_i\). For `l = 0`, `project!` may return 0.

`idx` is the list of groups (index vectors or ranges). The sets must be disjoint.
`prox!(y, f, x, γ)` writes the proximal point of \(h\) at `x` into `y`, and a
coordinate in no group is outside \(h\) so it stays equal to `x`.

`λ` is a scalar or a vector. `group_lambda` stores it as a float vector.
A scalar is used for every group. A vector has one weight per group.

`GroupRankProjL0(λ, n, l, s, project!)` passes an integer instead of `idx`.
`l`, `s`, and `project!` are unchanged, and the groups are `[1:1, 2:2, ..., n:n]`.
A vector `λ` then has length `n`.

`project!(dst, src, i)` writes the Euclidean projection of group `i` onto \(D_i\)
into `dst`. That vector must have the same length as the corresponding group.
For `l > 0` it must not be the origin. For `l = 0`, `project!` may return 0.
A returned array must have that same length; every group that differs is reported.
A group turned off by the support is written as zero
afterwards and is not the origin check.

Shifted gains are \(\rho_i = \omega_i - 2\gamma\lambda_i\), with
\(\omega_i = \|x_{G_i}\|^2 - \|x_{G_i} - P_{D_i}(x_{G_i})\|^2\),
which are sorted and then ranked against zero by the filter in `groupL0support.jl`.

`l = 0`, `s = m` keeps a group when \(\rho_i > 0\) and does not sort.
"""
struct GroupRankProjL0{R <: Real, RR <: AbstractVector{R}, I, F}
  lambda::RR
  idx::I
  l::Int
  s::Int
  project!::F
  rho::Vector{R}
  perm::Vector{Int}
  proj::Vector{R}
  proj_at::Vector{Int}

  function GroupRankProjL0{R, RR, I, F}(
    lambda::RR,
    idx::I,
    l::Int,
    s::Int,
    project!::F,
  ) where {R <: Real, RR <: AbstractVector{R}, I, F}
    m = length(idx)
    check_groups_disjoint(idx)
    length(lambda) == m || error("number of weights and groups must be the same")
    check_lambda_nonnegative(lambda)
    at, nproj = proj_layout(idx)
    new{R, RR, I, F}(
      lambda,
      idx,
      l,
      s,
      project!,
      Vector{R}(undef, m),
      collect(1:m),
      Vector{R}(undef, nproj),
      at,
    )
  end
end

function GroupRankProjL0(
  lambda::Union{Real, AbstractVector{<:Real}},
  idx,
  l::Integer,
  s::Integer,
  project!,
)
  lambda_v = group_lambda(lambda, length(idx))
  R = eltype(lambda_v)
  ll, ss = group_window(l, s, length(idx))
  GroupRankProjL0{R, typeof(lambda_v), typeof(idx), typeof(project!)}(lambda_v, idx, ll, ss, project!)
end

# GroupRankProjL0(λ, n, l, s, project!) passes an integer instead of `idx`.
# The groups are `[1:1, 2:2, ..., n:n]`. A vector `λ` then has length `n`.
function GroupRankProjL0(
  lambda::Union{Real, AbstractVector{<:Real}},
  n::Integer,
  l::Integer,
  s::Integer,
  project!,
)
  GroupRankProjL0(lambda, coordinate_groups(n), l, s, project!)
end

function (f::GroupRankProjL0)(x::AbstractArray{R}) where {R <: Real}
  # Membership of a nonzero block in Dᵢ is not rechecked: the prox writes
  # either 0 or the stored projection. The cardinality window is checked.
  val, nact = group_penalty(f.lambda, f.idx, x)
  if nact < f.l || nact > f.s
    return typeof(val)(Inf)
  end
  return val
end

function prox!(
  y::AbstractArray{R},
  f::GroupRankProjL0{R},
  x::AbstractArray{R},
  γ::R = one(R),
) where {R <: Real}
  # y receives prox_{γh}(x). Coordinates in no group stay equal to x.
  fill_proj_gains!(f.rho, f.proj, f.proj_at, x, f.idx, f.lambda, γ, f.project!, f.l > 0)
  k = group_l0_support!(f.perm, f.rho, f.l, f.s)
  return write_group_support!(y, x, f.idx, f.lambda, f.perm, k, f.proj, f.proj_at)
end

fun_name(f::GroupRankProjL0) = "group ℓ₀ penalty with cardinality and sets"
fun_dom(f::GroupRankProjL0) = "AbstractArray{<:Real}"
fun_expr(f::GroupRankProjL0) = "x ↦ Σᵢ λᵢ ‖x_{Gᵢ}‖₀ + χ_{B ∩ C_{l,s}}(x)"
fun_params(f::GroupRankProjL0) = "m = $(length(f.lambda)), l = $(f.l), s = $(f.s)"
