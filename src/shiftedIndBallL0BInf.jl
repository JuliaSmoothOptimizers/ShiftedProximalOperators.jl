export ShiftedIndBallL0BInf

function fill_large!(large::AbstractVector{Int}, x::AbstractVector{R}, Δ::R) where R
  # the first segment of large are the large components of xk
  # the rest is its complement
  n = length(x)
  length(large) == n || error("input vectors must have same length")
  i = 1
  j = n
  for k = 1 : n
    if abs(x[k]) > Δ
      large[i] = k
      i += 1
    else
      large[j] = k
      j -= 1
    end
  end
  nlarge = i - 1
  return nlarge
end

mutable struct ShiftedIndBallL0BInf{
  I <: Integer,
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3 <: AbstractVector{R}
} <: ShiftedProximableFunction
  h::IndBallL0{I}
  xk::V0
  sj::V1
  sol::V2
  z::V3
  p::Vector{Int}
  Δ::R
  χ::Conjugate{IndBallL1{R}}
  shifted_twice::Bool
  large::Vector{Int}
  nlarge::Int

  function ShiftedIndBallL0BInf(
    h::IndBallL0{I},
    xk::AbstractArray{R},
    sj::AbstractVector{R},
    Δ::R,
    χ::Conjugate{IndBallL1{R}},
    shifted_twice::Bool
  ) where {I <: Integer, R <: Real}
    sol = similar(sj)
    z = similar(sj)
    n = length(xk)
    nz = 0
    for i = 1 : n
      if xk[i] != 0
        nz += 1
      end
    end
    nz ≤ h.r || error("center of trust region not in $(h.r)B₀")
    large = Vector{Int}(undef, n)
    nlarge = fill_large!(large, xk, Δ)

    new{I, R, typeof(xk), typeof(sj), typeof(sol), typeof(z)}(
      h,
      xk,
      sj,
      sol,
      z,
      Vector{Int}(undef, length(sj)),
      Δ,
      χ,
      shifted_twice,
      large,
      nlarge
    )
  end
end

(ψ::ShiftedIndBallL0BInf)(y) = begin
  hval = ψ.h(ψ.xk + ψ.sj + y)
  indval = IndBallLinf(1.1 * ψ.Δ)(ψ.sj + y)
  val = hval + indval
  # @info "" hval indval
  if isinf(val)
    @error "nonsmooth model returns infinite value" ψ y hval indval
  end
  return hval  # FIXME
end

# need to adjust some index sets when the first shift is updated
function shift!(ψ::ShiftedIndBallL0BInf, shift::AbstractVector{R}) where {R <: Real}
  if ψ.shifted_twice
    ψ.sj .= shift
  else
    ψ.xk .= shift
    ψ.nlarge = fill_large!(ψ.large, ψ.xk, ψ.Δ)
  end
  return ψ
end

# need to adjust some index sets when the radius is updated
function set_radius!(ψ::ShiftedIndBallL0BInf, Δ::R) where {R <: Real}
  ψ.Δ = Δ
  ψ.nlarge = fill_large!(ψ.large, ψ.xk, ψ.Δ)
  return ψ
end

shifted(
  h::IndBallL0{I},
  xk::AbstractVector{R},
  Δ::R,
  χ::Conjugate{IndBallL1{R}},
) where {I <: Integer, R <: Real} = ShiftedIndBallL0BInf(h, xk, zero(xk), Δ, χ, false)

shifted(
  ψ::ShiftedIndBallL0BInf{I, R, V0, V1, V2},
  sj::AbstractVector{R},
) where {
  I <: Integer,
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R}
} = ShiftedIndBallL0BInf(ψ.h, ψ.xk, sj, ψ.Δ, ψ.χ, true)

fun_name(ψ::ShiftedIndBallL0BInf) = "shifted L0 norm ball with L∞-norm trust region indicator"
fun_expr(ψ::ShiftedIndBallL0BInf) = "t ↦ χ({‖xk + sj + t‖₀ ≤ r}) + χ({‖sj + t‖∞ ≤ Δ})"
fun_params(ψ::ShiftedIndBallL0BInf) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "Δ = $(ψ.Δ)"

# helper function
# projection into the intersection of x + ΔΒ∞ with a "piece" of rB₀ defined by the indices in S: A{S} := {w | w[Sᶜ] = 0}
function projB0!(w::AbstractVector{R}, S, x::AbstractVector{R}, Δ::R) where R
  for i = 1 : length(w)
    if i ∈ S
      w[i] = min(x[i] + Δ, max(x[i] - Δ, w[i]))
    else
      w[i] = 0
    end
  end
  w
end

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedIndBallL0BInf{I, R, V0, V1, V2, V3},
  q::AbstractVector{R},
  σ::R,
) where {
  I <: Integer,
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3 <: AbstractVector{R}
}
  @debug "entering prox! with" ψ q' σ
  r = ψ.h.r

  # special cases that we get out of the way
  if r == 0
    @debug "r = 0 ⟹ solution is zero"
    # 0B₀ = {0} and x == 0
    y .= -(ψ.xk .+ ψ.sj)
    return y
  end

  y .= ψ.xk .+ ψ.sj .+ q

  if all(ψ.xk .- ψ.Δ .≤ y .≤ ψ.xk .+ ψ.Δ)
    @debug "w is already in the box, simply project into $(ψ.h.r)B₀"
    sortperm!(ψ.p, y, rev = true, by = abs)
    y[ψ.p[(ψ.h.r + 1):end]] .= 0 # set smallest to zero
    y .-= ψ.xk .+ ψ.sj
    # @show "returning" y ψ(y)
    return y
  end

  if ψ.nlarge == r
    @debug "intersection only concerns supp(x)"
    # rB₀ ∩ (x + Δ B∞) = supp(x) ∩ (x + ΔΒ∞)
    # y = proj(q | supp(x) ∩ (x + ΔΒ∞)) - (x + s)
    for i ∈ eachindex(y)
      if ψ.xk[i] == 0
        y[i] = -(ψ.xk[i] + ψ.sj[i])
      else
        # y = min(x + Δ, max(x - Δ, y)) - (x + s)
        y[i] = min(-ψ.sj[i] + ψ.Δ, max(-ψ.sj[i] - ψ.Δ, q[i]))
      end
    end
    return y
  end

  n = length(ψ.xk)
  if r == n
    @debug "r = n ⟹ solution is projection of y into box"
    # nB₀ = Rⁿ
    # y <- proj(y | x + ΔΒ∞)
    for i ∈ eachindex(y)
      y[i] = min(-ψ.sj[i] + ψ.Δ, max(-ψ.sj[i] - ψ.Δ, q[i]))
    end
    return y
  end

  @debug "general case"
  nlarge = ψ.nlarge
  ψ.z .= y  # use z as temporary storage (this is the vector we want to project)
  y .= min.(ψ.xk .+ ψ.Δ, max.(ψ.xk .- ψ.Δ, y))  # y <- proj(y | x + ΔΒ)
  large = view(ψ.large, 1 : nlarge)
  F = view(ψ.large, (nlarge + 1) : n)
  @views ψ.z[F] .= ψ.z[F].^2 .- (ψ.z[F] .- y[F]).^2
  # sort components of z in descending order
  # @views sortperm!(ψ.p[1 : (n - nlarge)], ψ.z[F], rev = true)
  # p = view(ψ.p, 1 : (r - nlarge))  # z[F[q]] are the r - nlarge largest components of z[F]
  # @views projB0!(y, (i for i = 1 : n if (i ∈ ψ.large) || (i ∈ F[p])), ψ.xk, ψ.Δ)
  p = sortperm(ψ.z[F], rev = true)
  p2 = view(p, 1 : (r - nlarge))
  projB0!(y, (i for i = 1 : n if (i ∈ large || i ∈ F[p2])), ψ.xk, ψ.Δ)
  # if nlarge > 0
  #   @show nlarge large F ψ.z[F] p p2
  # end

  y .-= ψ.xk .+ ψ.sj
  # @show y
  return y
end


# F = setdiff(1:n, large)
# z = similar(w)
# @views z[F] = w[F].^2 - (w[F] - pw[F]).^2
# p = Vector{Int}(undef, length(w))
# @views sortperm!(p[1 : (n - nlarge)], z[F], rev = true)
# q = view(p, 1 : (r - nlarge))
# projB0!(pw, (i for i = 1 : n if (i ∈ large) || (i ∈ view(F, q))), x, Δ)