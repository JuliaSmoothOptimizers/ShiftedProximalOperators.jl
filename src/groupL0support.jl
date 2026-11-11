# Shared proximal kernel for the group-ℓ₀ maps.
#
# Shifted gains are ρᵢ = ωᵢ - 2γλᵢ, compared with zero. The caller fills `rho`.
# `perm` is a workspace of length m = length(rho). It is allocated as 1:m and
# stays a permutation of the group indices. On return, perm[1:k] are the
# selected ones and the return value is k.
#
# Rule, for one proximal support. Gains are compared with zero.
# s = m does not discard groups. One pass keeps every strict positive and
# records the zeros. If the positives are already at least l, stop. Otherwise
# add the smallest number of zeros that reaches l. If the zeros are not
# enough, sort the (l − current) largest negatives and add them. l = 0 stops
# after the positives: nothing is sorted.
# l = 0 and s < m looks only at strict positives. If there are at most s, keep
# them. Otherwise drop the smallest (number of positives − s) of them.
# 0 < l ≤ s < m places the l largest first. If the l-th is ≤ 0, stop. If it is
# > 0, bring the remaining strict positives forward and keep them when they fit
# in s; otherwise select the largest s − l of that block. Nonpositives past the
# l-th are never sorted.

"""
    coordinate_groups(n)

One group per coordinate, `[1:1, 2:2, …, n:n]`.
"""
function coordinate_groups(n::Integer)
  n >= 0 || error("number of coordinates must be nonnegative")
  return [i:i for i in 1:Int(n)]
end

# Group ℓ₀ penalties. 
# Fill a vector with the same value for every group.
# `1` and `[1, 2]` must still apply to a Float64 vector.
function group_lambda(lambda::R, m::Integer) where {R <: Real}
  return fill(float(lambda), Int(m))
end

function group_lambda(lambda::AbstractVector{<:Real}, m::Integer)
  length(lambda) == m || error("number of weights and groups must be the same")
  return float.(lambda)
end

# Names every negative entry. The word index is written once.
function check_lambda_nonnegative(lambda::AbstractVector)
  neg = Int[]
  for (i, λ) in pairs(lambda)
    λ < 0 && push!(neg, Int(i))
  end
  isempty(neg) && return lambda
  error("weights λ must be nonnegative: negative at index $(join(neg, ", "))")
end

# Cardinality window
function group_window(l::Integer, s::Integer, m::Integer)
  if l < 0 || s < l || s > m
    error("cardinality window must satisfy 0 ≤ l ≤ s ≤ total number of groups")
  end
  return Int(l), Int(s)
end

# A coordinate in two groups could be zeroed and then overwritten by whichever
# selected group is written last, or vice versa, and the penalty would count it twice.
function check_groups_disjoint(idx)
  seen = Dict{Int, Int}()
  for (i, g) in enumerate(idx)
    for j in g
      jj = Int(j)
      p = get(seen, jj, 0)
      if p == i
        error("group $i repeats index $jj")
      elseif p != 0
        error("groups must be disjoint: index $jj belongs to groups $p and $i")
      end
      seen[jj] = i
    end
  end
  return idx
end

# proj is packed: group i occupies proj[at[i] : at[i] + length(idx[i]) − 1].
# `at === nothing` means proj is indexed by the coordinates of x, which is the
# l = 0 path where the candidate block is x itself.
function proj_layout(idx)
  m = length(idx)
  at = Vector{Int}(undef, m)
  n = 0
  for i in 1:m
    at[i] = n + 1
    n += length(idx[i])
  end
  return at, n
end

function group_block(proj, at, idx, i)
  at === nothing && return view(proj, idx[i])
  a = at[i]
  return view(proj, a:(a + length(idx[i]) - 1))
end

# Median of the three positions. Afterward the largest gain is at a and the smallest at c.
function median3!(perm::Vector{Int}, gains::Vector{R}, a::Int, b::Int, c::Int) where {R <: Real}
  @inbounds begin
    if gains[perm[a]] < gains[perm[b]]
      perm[a], perm[b] = perm[b], perm[a]
    end
    if gains[perm[b]] < gains[perm[c]]
      perm[b], perm[c] = perm[c], perm[b]
    end
    if gains[perm[a]] < gains[perm[b]]
      perm[a], perm[b] = perm[b], perm[a]
    end
  end
end

# Pivot at hi. Median of three, or Tukey's ninther when the window is longer than 128.
# The ninther is Bentley and McIlroy, Software—Practice and Experience 23 (1993).
# The size 128 is the one in pdqsort (Peters).
function place_pivot!(perm::Vector{Int}, gains::Vector{R}, lo::Int, hi::Int) where {R <: Real}
  len = hi - lo + 1
  mid = lo + (len >>> 1)
  @inbounds if len > 128
    median3!(perm, gains, lo, hi, mid)
    median3!(perm, gains, lo + 1, mid - 1, hi - 1)
    median3!(perm, gains, lo + 2, mid + 1, hi - 2)
    median3!(perm, gains, mid - 1, hi, mid + 1)
  else
    median3!(perm, gains, mid, hi, lo)
  end
end

# Pivot sits at hi. Strictly larger gains move left. Equals stay on the right.
function partition_above_pivot!(perm::Vector{Int}, gains::Vector{R}, lo::Int, hi::Int) where {R <: Real}
  @inbounds begin
    pivot = gains[perm[hi]]
    i = lo
    for j in lo:(hi - 1)
      if gains[perm[j]] > pivot
        perm[i], perm[j] = perm[j], perm[i]
        i += 1
      end
    end
    perm[i], perm[hi] = perm[hi], perm[i]
    return i
  end
end

# Move every gain equal to `pivot` to the front of lo:hi.
# Returns the first index after that run.
function gather_equal!(perm::Vector{Int}, gains::Vector{R}, lo::Int, hi::Int, pivot::R) where {R <: Real}
  i = lo
  @inbounds for j in lo:hi
    if gains[perm[j]] == pivot
      perm[i], perm[j] = perm[j], perm[i]
      i += 1
    end
  end
  return i
end

# Quarter-point swaps inside lo:hi, after a cut that threw away less than one
# eighth of the window. The fraction and the swaps are from pdqsort (Peters).
# Only the side that does not contain k is tested: a pivot near k is kept.
function shuffle_window!(perm::Vector{Int}, lo::Int, hi::Int)
  len = hi - lo + 1
  len < 8 && return
  q = len ÷ 4
  @inbounds begin
    perm[lo], perm[lo + q] = perm[lo + q], perm[lo]
    perm[hi], perm[hi - q] = perm[hi - q], perm[hi]
    if len > 128
      perm[lo + 1], perm[lo + q + 1] = perm[lo + q + 1], perm[lo + 1]
      perm[lo + 2], perm[lo + q + 2] = perm[lo + q + 2], perm[lo + 2]
      perm[hi - 1], perm[hi - q - 1] = perm[hi - q - 1], perm[hi - 1]
      perm[hi - 2], perm[hi - q - 2] = perm[hi - q - 2], perm[hi - 2]
    end
  end
end

# Place the cutoff-th largest of perm[lo:hi] at position cutoff (gains compared through perm).
# Entries lo:cutoff-1 are then ≥ that value, and entries cutoff+1:hi are ≤ it.
# perm holds group indices, not coordinates.
# A cut that throws away less than one eighth of the window is followed by the
# quarter swaps above. If the pivot equals the entry just outside the window,
# every copy of that value is gathered and the search steps past the run.
# That test is pdqsort's (Peters): the equal block is not partitioned again.
function select_largest_through!(perm::Vector{Int}, gains::Vector{R}, lo::Int, cutoff::Int, hi::Int) where {R <: Real}
  leftmost = true
  while lo < hi
    if !leftmost && gains[perm[lo]] == gains[perm[lo - 1]]
      nxt = gather_equal!(perm, gains, lo, hi, gains[perm[lo - 1]])
      cutoff < nxt && return perm
      lo = nxt
      continue
    end
    place_pivot!(perm, gains, lo, hi)
    if !leftmost && gains[perm[hi]] == gains[perm[lo - 1]]
      nxt = gather_equal!(perm, gains, lo, hi, gains[perm[hi]])
      cutoff < nxt && return perm
      lo = nxt
      continue
    end
    pvt = partition_above_pivot!(perm, gains, lo, hi)
    pvt == cutoff && return perm
    w = hi - lo + 1
    discarded = pvt < cutoff ? (pvt - lo + 1) : (hi - pvt + 1)
    bad = discarded * 8 < w
    if pvt < cutoff
      lo = pvt + 1
      leftmost = false
    else
      hi = pvt - 1
    end
    if bad && lo < hi
      shuffle_window!(perm, lo, hi)
    end
  end
  return perm
end

# s = m. perm[1:npos] positives, then the zeros, then the negatives.
# Returns the number of selected groups, written in perm[1:k].
function support_uncapped!(perm::Vector{Int}, rho::Vector{R}, l::Int) where {R <: Real}
  m = length(rho)
  lo = 1
  mid = 1
  hi = m
  # Swap positives to the front, zeros to the middle, and negatives to the back.
  @inbounds while mid <= hi
    g = rho[perm[mid]]
    if g > 0
      perm[lo], perm[mid] = perm[mid], perm[lo]
      lo += 1
      mid += 1
    elseif g < 0
      perm[mid], perm[hi] = perm[hi], perm[mid]
      hi -= 1
    else
      mid += 1
    end
  end
  # npos is the last positive index
  npos = lo - 1
  # l = 0 keeps a group only when its shifted gain is positive.
  npos >= l && return npos
  # hi is the last nonnegative index, so 1:hi are the positives and the zeros.
  # We return l (smallest needed amount of groups with arbitrary choice of zeros)
  hi >= l && return l
  # Pivot partition exactly l − (positives and zeros) of the negatives, and add those.
  need = l - hi
  select_largest_through!(perm, rho, hi + 1, hi + need, m)
  return hi + need
end

# l = 0 and s < m. Only strict positives are eligible, and at most s of them.
# Swap them to the front. The tail keeps the other groups, each index once.
function support_positive_cap!(perm::Vector{Int}, rho::Vector{R}, s::Int) where {R <: Real}
  m = length(rho)
  npos = 0
  @inbounds for i in 1:m
    if rho[perm[i]] > 0
      npos += 1
      perm[npos], perm[i] = perm[i], perm[npos]
    end
  end
  npos <= s && return npos
  # Drop the npos − s smallest positives. The s largest stay in front.
  select_largest_through!(perm, rho, 1, s, npos)
  return s
end

function group_l0_support!(perm::Vector{Int}, rho::Vector{R}, l::Int, s::Int) where {R <: Real}
  m = length(rho)
  # s = 0 leaves every group inactive. l ≤ s, so l is 0 as well.
  s == 0 && return 0
  s == m && return support_uncapped!(perm, rho, l)
  l == 0 && return support_positive_cap!(perm, rho, s)
  # Sort the largest l groups by shifted gain.
  select_largest_through!(perm, rho, 1, l, m)
  # The l-th is kept when it only ties the threshold.
  if rho[perm[l]] <= 0 || l == s
    return l
  end
  # One pass: strict positives in front, nonpositives ignored from here on.
  i = l + 1
  @inbounds for j in (l + 1):m
    if rho[perm[j]] > 0
      perm[i], perm[j] = perm[j], perm[i]
      i += 1
    end
  end
  # Number of positives in front of l
  npos = i - (l + 1)
  # If the largest l groups plus the next npos fit in s, return l + npos.
  l + npos <= s && return l + npos
  # Otherwise, pivot partition the next npos groups and return s.
  select_largest_through!(perm, rho, l + 1, s, l + npos)
  return s
end

# Whether any coordinate in group g is nonzero, x_{Gᵢ} ≠ 0.
function group_active(x, g)
  xg = view(x, g)
  @inbounds for t in eachindex(xg)
    xg[t] != 0 && return true
  end
  return false
end

# Set every coordinate in the selected groups to zero.
function zero_groups!(y, idx)
  for g in idx
    yg = view(y, g)
    @inbounds for t in eachindex(yg)
      yg[t] = 0
    end
  end
  return y
end

# Write the proximal point from the chosen support `perm[1:cutoff]`.
# The copy plants x on every coordinate. A coordinate in no group is outside h
# and is left there. Every group block is then cleared, and each selected block
# is replaced by its projection. `proj` is packed by group, unless `at` is
# nothing, in which case it is indexed like `x`. A selected block that is still
# identically zero is not charged. A projection equal to the origin is rejected
# before this copy.
function write_group_support!(y::AbstractArray{R}, x, idx, lambda, perm, cutoff::Int, proj, at) where {R <: Real}
  copyto!(y, x) # a coordinate in no group is outside h and stays equal to x
  zero_groups!(y, idx)
  penalty = zero(R)
  @inbounds for t in 1:cutoff
    group = perm[t]
    indices = idx[group]
    y_block = view(y, indices)
    proj_block = group_block(proj, at, idx, group)
    active = false
    for j in eachindex(y_block)
      y_block[j] = proj_block[j]
      active |= y_block[j] != 0
    end
    active && (penalty += lambda[group])
  end
  return penalty
end

# Group hard threshold: keep group i when ‖x_{Gᵢ}‖² > 2γλᵢ.
# Used by GroupNormL0. GroupRankL0 goes through the ball projection instead.
# The copy plants x on every coordinate. A coordinate in no group is outside h
# and is left there. A rejected group is zeroed; a kept group is already x.
function group_hard_threshold!(y::AbstractArray{R}, x::AbstractArray{R}, idx, lambda, γ::R) where {R <: Real}
  copyto!(y, x) # a coordinate in no group is outside h and stays equal to x
  scale = 2 * γ
  val = zero(R)
  for (g, λ) in zip(idx, lambda)
    xg = view(x, g)
    nrm2 = zero(R)
    @inbounds for t in eachindex(xg)
      nrm2 += xg[t] * xg[t]
    end
    if nrm2 > scale * λ
      val += λ
    else
      yg = view(y, g)
      @inbounds for t in eachindex(yg)
        yg[t] = 0
      end
    end
  end
  return val
end

# ∑ᵢ₌₁ᵐ λᵢ ‖x_{Gᵢ}‖₀
function group_penalty(lambda, idx, x)
  R = real(eltype(x))
  val = zero(R)
  nact = 0
  for (g, λ) in zip(idx, lambda)
    if group_active(x, g)
      val += λ
      nact += 1
    end
  end
  return val, nact
end

# D = {z : ‖z‖ ≥ δ}. Outside the ball the projection is x. Inside, it is the
# radial point on the sphere. At the origin every point of the sphere is nearest;
# the first coordinate in the group's order is set to +δ and the rest to 0.
# Returns whether the block was strictly inside the ball.
function project_outside_ball!(proj_g, xg, δ::R, i::Integer) where {R <: Real}
  nrm2 = zero(R)
  @inbounds for t in eachindex(xg)
    nrm2 += xg[t] * xg[t]
  end
  nrm = sqrt(nrm2)
  if nrm >= δ
    @inbounds for t in eachindex(xg)
      proj_g[t] = xg[t]
    end
    return false
  end
  if nrm == 0
    t0 = iterate(eachindex(proj_g))
    t0 === nothing && error("group $i is empty")
    @inbounds for t in eachindex(proj_g)
      proj_g[t] = 0
    end
    proj_g[t0[1]] = δ
    return true
  end
  scale = δ / nrm
  @inbounds for t in eachindex(xg)
    proj_g[t] = xg[t] * scale
  end
  return true
end

# Dᵢ = ℝ^{nᵢ}, so P = x and ρᵢ = ‖x_{Gᵢ}‖² - 2γλᵢ. Used when l = 0  in GroupRankL0.
function fill_copy_gains!(rho::Vector{R}, x, idx, lambda, γ::R) where {R <: Real}
  scale = 2 * γ
  for (i, (g, λ)) in enumerate(zip(idx, lambda))
    xg = view(x, g)
    nrm2 = zero(R)
    @inbounds for t in eachindex(xg)
      nrm2 += xg[t] * xg[t]
    end
    rho[i] = nrm2 - scale * λ
  end
  return rho
end

# Shifted gains for the exclusion ball. `snapped[i]` is true when group i was
# strictly inside the ball. The radius is compared with the square root: δ²
# underflows once δ drops below sqrt(floatmin), while δ itself is still representable.
function fill_ball_gains!(rho::Vector{R}, proj, at, snapped, x, idx, lambda, γ::R, δ::R) where {R <: Real}
  scaleγ = 2 * γ
  for (i, (g, λ)) in enumerate(zip(idx, lambda))
    xg = view(x, g)
    proj_g = group_block(proj, at, idx, i)
    snapped[i] = project_outside_ball!(proj_g, xg, δ, i)
    xx = zero(R)
    dist = zero(R)
    @inbounds for t in eachindex(xg)
      xx += xg[t] * xg[t]
      d = xg[t] - proj_g[t]
      dist += d * d
    end
    rho[i] = xx - dist - scaleγ * λ
  end
  return rho
end

# For l > 0 the vector just written by project! must be a point of Dᵢ, hence not the origin.
# For l = 0, project! may return 0, so the caller skips this check.
# A group the support turns off is set to zero later, in the result, and is not inspected here.
function projection_is_origin(proj_g)
  for t in eachindex(proj_g)
    proj_g[t] != 0 && return false
  end
  return true
end

function projection_origin_error(groups)
  parts = ["group $i" for i in groups]
  error("project! returned the origin for " * join(parts, ", ") * "; 0 must not lie in Dᵢ")
end

# A returned array is the projection, and its length must be the group's.
# DimensionMismatch or BoundsError means the in-place write already had the wrong length.
function note_projection_mismatch(bad, i::Int, nidx::Int, nproj::Union{Nothing,Int})
  bad === nothing && (bad = Tuple{Int,Int,Union{Nothing,Int}}[])
  push!(bad, (i, nidx, nproj))
  return bad
end

function projection_length_error(bad)
  parts = String[]
  for (i, nidx, nproj) in bad
    if nproj === nothing
      push!(parts, "group $i has idx length $nidx and the projection length differs")
    else
      push!(parts, "group $i has idx length $nidx and projection length $nproj")
    end
  end
  error("project! length does not match idx: " * join(parts, ", "))
end

# ωᵢ = ‖x‖² - ‖x - P_{Dᵢ}(x)‖², then ρᵢ = ωᵢ - 2γλᵢ.
# `project!(dst, src, i)` writes P_{Dᵢ}(src) into dst. The projection stays in `proj` for later use.
# Every group whose projection length differs from its idx entry is reported together.
# Every group whose projection is the origin is reported together as well.
# A coordinate in no group is outside h and is left there.
function fill_proj_gains!(rho::Vector{R}, proj, at, x, idx, lambda, γ::R, project!, check_origin::Bool) where {R <: Real}
  # Every group whose projection length differs from its idx entry is reported together.
  bad = nothing
  for (i, g) in enumerate(idx)
    xg = view(x, g)
    proj_g = group_block(proj, at, idx, i)
    returned = try
      project!(proj_g, xg, i)
    catch err
      if !(err isa DimensionMismatch || err isa BoundsError)
        rethrow()
      end
      bad = note_projection_mismatch(bad, i, length(xg), nothing)
      continue
    end
    if returned isa AbstractArray && length(returned) != length(xg)
      bad = note_projection_mismatch(bad, i, length(xg), length(returned))
    elseif returned isa AbstractArray && !(returned === proj_g)
      copyto!(proj_g, returned)
    end
  end
  bad === nothing || projection_length_error(bad)
  # Every group whose projection is the origin is reported together as well.
  if check_origin
    origins = Int[]
    for i in eachindex(idx)
      projection_is_origin(group_block(proj, at, idx, i)) && push!(origins, i)
    end
    isempty(origins) || projection_origin_error(origins)
  end
  # Fill the gains.
  scale = 2 * γ
  for (i, (g, λ)) in enumerate(zip(idx, lambda))
    xg = view(x, g)
    proj_g = group_block(proj, at, idx, i)
    xx = zero(R)
    dist = zero(R)
    @inbounds for t in eachindex(xg)
      xx += xg[t] * xg[t]
      d = xg[t] - proj_g[t]
      dist += d * d
    end
    rho[i] = xx - dist - scale * λ
  end
  return rho
end
