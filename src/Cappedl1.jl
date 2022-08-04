# Cappedl1 function
export Cappedl1


mutable struct Cappedl1{R <: Real, S <: AbstractArray, T, Tr, M <: AbstractArray{T}}
  lambda::R
  theta::R
  A::S
  F::PSVD{T, Tr, M}
  function Cappedl1(
    lambda::R,
    theta::R,
    A::S,
    F::PSVD{T, Tr, M},
  ) where {R <: Real, S <: AbstractArray, T, Tr, M <: AbstractArray{T}}
    if lambda < 0 || theta < 0
      error("λ must be nonnegative")
    end
    new{typeof(lambda),  typeof(A), T, Tr, M}(lambda, theta, A, F)
  end
end

Cappedl1(lambda::R, theta::R, A::S, F::PSVD{T, Tr, M}) where {R, S, T, Tr, M <: AbstractArray{T}} =
  Cappedl1{R, S, eltype(A), real(eltype(A)), M}(lambda, theta, A, F)

Cappedl1(lambda::R, theta::R, A::S) where {R, S} = begin
  F = psvd_workspace_dd(A, full = false)
  Cappedl1(lambda, theta, A, F)
end
  
function (f::Cappedl1)(x::AbstractVector{R}) where {R <: Real}
    f.A .= ShiftedProximalOperators.reshape_array(x, size(f.A))
    psvd_dd!(f.F, f.A, full = false)
    return f.lambda * sum(min.(f.F.S, f.theta))
end

fun_name(f::Cappedl1) = "Cappedl1"
fun_dom(f::Cappedl1) = "AbstractArray{Real}"
fun_expr(f::Cappedl1{T}) where {T <: Real} = "x ↦ Cappedl1(matrix(x))"
fun_params(f::Cappedl1{T}) where {T <: Real} = "λ = $(f.lambda)"

function prox!(
  y::AbstractVector{R},
  f::Cappedl1{R, S, T, Tr, M},
  x::AbstractVector{R},
  gamma::R,
) where {R <: Real, S <: AbstractArray, T, Tr, M <: AbstractArray{T}}
  f.A .= ShiftedProximalOperators.reshape_array(x, size(f.A))
  psvd_dd!(f.F, f.A, full = false)
#  c = sqrt(2 * f.lambda * gamma)
#  f.F.S .= max.(0, f.F.S .- f.lambda*gamma)
  for i ∈ eachindex(f.F.S)
    x1 = max(f.theta, f.F.S[i])
    x2 = min(f.theta, max(0, f.F.S[i] - f.lambda * gamma))
    if 0.5*(x1 - f.F.S[i])^2 + f.lambda * gamma * f.theta < 0.5*(x2 - f.F.S[i])^2 + f.lambda * gamma * x2
        f.F.S[i] = x1
    else
        f.F.S[i] = x2
    end
    for j = 1:size(f.A, 1)
        f.F.U[j, i] = f.F.U[j, i] * f.F.S[i]
    end
  end
  mul!(f.A, f.F.U, f.F.Vt)
  y .= ShiftedProximalOperators.reshape_array(f.A, (size(y, 1), 1))
  return y
end
