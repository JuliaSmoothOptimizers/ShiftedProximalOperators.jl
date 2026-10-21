export NormL2Inf

"""
    NormL2Inf(ω)

Weighted squared ℓ₂,∞ norm defined on ℝⁿᴺ:

    f(x) = max_{1 ≤ i ≤ N} ωᵢ ‖xᵢ‖²

where x = (x₁, …, xN) with xᵢ ∈ ℝⁿ and ωᵢ > 0.

The groups are contiguous blocks of equal size n = div(length(x), N), so that
xᵢ = x[((i-1)*n + 1) : i*n].

# Fields
- `omega::Vector{R}`: positive weights ωᵢ, one per group (length N).
- `n::Int`: size of each group (set at first call if 0, or given explicitly).

# Reference
Propositions 2.1 and 2.2 in:
  "Proximity operator of the weighted squared ℓ₂,∞ norm with applications",
  S. López-Rivera, 2026.
"""
mutable struct NormL2Inf{R <: Real} # <: ProximableFunction
  omega::Vector{R} # weights ωᵢ > 0, length N
  n::Int # group size (number of components per block)
  α::Vector{R} # αᵢ = ωᵢ ‖xᵢ‖²
  λv::Vector{R} # λᵢ = λ ωᵢ
  p::Vector{R} # solution simplex
  order::Vector{Int} # buffer
  α_sqrt::Vector{R} # buffer
  λ_sqrt::Vector{R} # buffer

  function NormL2Inf(omega::AbstractVector{R}, n::Int = 0) where {R <: Real}
    all(>(0), omega) || error("All weights ωᵢ must be strictly positive.")
    n >= 0 || error("Group size n must be non-negative.")
    N = length(omega)
    new{R}(
      collect(omega), n, Vector{R}(undef, N), Vector{R}(undef, N), Vector{R}(undef, N),
      Vector{Int}(undef, N), Vector{R}(undef, N), Vector{R}(undef, N),
      )
  end
end

# Scalar convenience: N groups of size n, all with the same weight ω
NormL2Inf(omega::R, N::Int, n::Int) where {R <: Real} =
  NormL2Inf(fill(omega, N), n)

"""
    _group_size(h, x)

Return the group size n, checking consistency with `length(x)`.
If `h.n == 0` the size is inferred from `length(x)` and `length(h.omega)`.
"""
function _group_size(h::NormL2Inf, x::AbstractVector)
  N = length(h.omega)
  if h.n == 0
    rem(length(x), N) == 0 ||
      error("length(x) = $(length(x)) is not divisible by N = $N.")
    return div(length(x), N)
  else
    length(x) == N * h.n ||
      error("length(x) = $(length(x)) ≠ N × n = $(N * h.n).")
    return h.n
  end
end

"""
    _group_range(i, n)

Return the index range of the i-th group (1-based) for block size n.
"""
_group_range(i::Int, n::Int) = ((i - 1) * n + 1):(i * n)


function (h::NormL2Inf{R})(x::AbstractVector{R}) where {R <: Real}
  n = _group_size(h, x)
  N = length(h.omega)
  val = zero(R)
  for i = 1:N
    xi = view(x, _group_range(i, n))
    val = max(val, h.omega[i] * sum(abs2, xi))
  end
  return val
end


"""
    _solve_aux_simplex(α, λv)

Solve the auxiliary concave simplex problem:

    max_{p ∈ ΔN}  ∑ᵢ αᵢ pᵢ / (1 + 2 λᵢ pᵢ)

explicitly via KKT conditions.

# Arguments
- `α::Vector{R}`: αᵢ = ωᵢ ‖xᵢ‖², length N.
- `λv::Vector{R}`: λᵢ = λ ωᵢ, length N.

# Returns
`p::Vector{R}`: the solution in ΔN.

See Proposition 2.2 of "Proximity operator of the weighted squared ℓ₂,∞ norm with applications", S. López-Rivera, 2026.
"""
function _solve_aux_simplex!(
    p,
    α::Vector{R},
    λv::Vector{R},
    order::Vector{Int},
    α_srt::Vector{R},
    λ_srt::Vector{R}
  ) where {R <: Real}
  N = length(α)
  # p = zeros(R, N)

  # If all αᵢ = 0, any simplex point is optimal
  if all(iszero, α)
    p .= one(R) / N
    return p
  end

  # Sort indices so that α_{ℓ₁} ≤ … ≤ α_{ℓN}
  # Initialize sums over the full active set {1,…,N}
  # S₁ = Σ_{j∉Iᵢ} 1/λⱼ, S₂ = Σ_{j∉Iᵢ} √αⱼ/λⱼ
  sortperm!(order, α)
  S1 = zero(R)
  S2 = zero(R)
  @inbounds for i = 1:N
    α_srt[i] = α[order[i]]
    λ_srt[i] = λv[order[i]]

    S1 += one(R) / λ_srt[i]
    S2 += sqrt(α_srt[i]) / λ_srt[i]
  end

  # Find k = min{ i ∈ {0,…,N-1} : (2 + S1) √α_{ℓᵢ₊₁} > S2 }
  # We remove one element at a time from the active set as i increases.
  k = N - 1   # N-1 is always in the set, so always valid
  for i = 0:(N - 2)
    next = i + 2
    if (2 + S1) * sqrt(α_srt[next]) > S2
      k = i
      break
    end
    # Remove sorted index `next` from active set
    S1 -= 1 / λ_srt[next]
    S2 -= sqrt(α_srt[next]) / λ_srt[next]
  end

  A = 2 + S1
  B = S2

  # Assign pᵢ for each original index
  for i = 1:N
    # position of original index i in the sorted order
    spos = 0
    for j = 1:N
      if order[j] == i
        spos = j
        break
      end
    end
    if spos <= k + 1
      # i ∈ Iₖ → pᵢ = 0
      p[i] = zero(R)
    else
      # i ∉ Iₖ → pᵢ = 1/(2λᵢ) * (A √αᵢ / B − 1)
      p[i] = (A * sqrt(α[i]) / B - one(R)) / (2 * λv[i])
    end
  end

  return p
end

"""
    prox!(y, h, x, λ)

Compute  y = prox_{λh}(x).

For each group i:  yᵢ = xᵢ / (1 + 2 λᵢ pᵢ),
where p solves the auxiliary simplex problem and λᵢ = λ ωᵢ.
"""
function prox!(
  y::AbstractVector{R},
  h::NormL2Inf{R},
  x::AbstractVector{R},
  λ::R,
) where {R <: Real}
  n = _group_size(h, x)
  N = length(h.omega)

  # Build αᵢ = ωᵢ ‖xᵢ‖²  and  λᵢ = λ ωᵢ
  for i = 1:N
    xi = view(x, _group_range(i, n))
    h.α[i] = h.omega[i] * norm(xi, 2)^2
    h.λv[i] = λ * h.omega[i]
  end

  # Solve auxiliary simplex problem
  _solve_aux_simplex!(h.p, h.α, h.λv, h.order, h.α_sqrt, h.λ_sqrt)

  # Apply prox formula and evaluate h(y)
  val = zero(R)
  for i = 1:N
    rng = _group_range(i, n)
    xi = view(x, rng)
    yi = view(y, rng)
    denom = one(R) + 2 * h.λv[i] * h.p[i]
    @. yi = xi / denom
    val = max(val, h.omega[i] * sum(abs2, yi))
  end

  return val
end