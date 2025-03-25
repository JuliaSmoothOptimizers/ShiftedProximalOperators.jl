export ShiftedCompositeNormL2
@doc raw"""
    ShiftedCompositeNormL2(h, c!, J!, A, b)

Returns the shift of a function `c` composed with the `ℓ₂` norm (see CompositeNormL2.jl).
Here, `c` is linearized i.e, `c(x + s) ≈ c(x) + J(x)s`. 
```math
f(s) = λ ‖c(x) + J(x)s‖₂,
```
where `λ > 0`. `c!` and `J!` should implement functions 
```math
c : ℝⁿ ↦ ℝᵐ,
```
```math
J : ℝⁿ ↦ ℝᵐˣⁿ,   
```
such that `J` is the Jacobian of `c`. It is expected that `m ≤ n`.
`A` and `b` should respectively be a matrix and a vector which can respectively store the values of `J` and `c`.
`A` is expected to be sparse, `c!` and `J!` should have signatures
```
c!(b <: AbstractVector{Real}, xk <: AbstractVector{Real})
J!(A <: AbstractSparseMatrixCOO{Real, Integer}, xk <: AbstractVector{Real})
```
"""
mutable struct ShiftedCompositeNormL2{
  T <: Real,
  F0 <: Function,
  F1 <: Function,
  M <: AbstractMatrix{T},
  V <: AbstractVector{T},
} <: ShiftedCompositeProximableFunction
  h::NormL2{T}
  c!::F0
  J!::F1
  A::M
  shifted_spmat::qrm_shifted_spmat{T}
  spfct::qrm_spfct{T}
  b::V
  g::V  # Preallocated vector used either to compute A*y + b when we call ψ(y) or the RHS of the dual of the proximal problem.
  q::V  # Preallocated solution vector of the dual of the proximal problem.
  dq::V # Preallocated vector to refine the q solution.
  p::V  # Preallocated vector used to compute s(α)ᵀ∇s(α) for the secular equation.
  dp::V # Preallocated vector used to refine the p vector.
  function ShiftedCompositeNormL2(
    λ::T,
    c!::Function,
    J!::Function,
    A::AbstractMatrix{T},
    b::AbstractVector{T},
  ) where {T <: Real}
    p = similar(b, A.n + A.m)
    dp = similar(b, A.n + A.m)
    g = similar(b)
    q = similar(b)
    dq = similar(b)
    if length(b) != size(A, 1)
      error("ShiftedCompositeNormL2: Wrong input dimensions, there should be as many constraints as rows in the Jacobian")
    end

    spmat = qrm_spmat_init(A; sym=false)
    shifted_spmat = qrm_shift_spmat(spmat)
    spfct = qrm_spfct_init(spmat)
  
    new{T, typeof(c!), typeof(J!), typeof(A), typeof(b)}(NormL2(λ), c!, J!, A, shifted_spmat, spfct, b, g, q, dq, p, dp)
  end
end
 
shifted(
  ψ::CompositeNormL2{T, F0, F1, M, V},
  xk::AbstractVector{T}
) where {T <: Real, F0 <: Function, F1 <: Function, M <: AbstractMatrix{T}, V <: AbstractVector{T}} = begin
  b = similar(ψ.b)
  ψ.c!(b, xk)
  A = similar(ψ.A)
  ψ.J!(A, xk)
  ShiftedCompositeNormL2(ψ.h.lambda, ψ.c!, ψ.J!, A, b)
end

fun_name(ψ::ShiftedCompositeNormL2) = "shifted `ℓ₂` norm"
fun_expr(ψ::ShiftedCompositeNormL2) = "t ↦ ‖c(xk) + J(xk)t‖₂"
fun_params(ψ::ShiftedCompositeNormL2) = "c(xk) = $(ψ.b)\n" * " "^14 * "J(xk) = $(ψ.A)\n"

function prox!(
  y::AbstractVector{T},
  ψ::ShiftedCompositeNormL2{T, F0, F1, M, V},
  q::AbstractVector{T},
  ν::T;
  max_iter = 10,
  atol = eps(T)^0.3,
  max_time = 180.0
) where {T <: Real, F0 <: Function, F1 <:Function, M <: AbstractMatrix{T}, V <: AbstractVector{T}}

  start_time = time()
  θ = T(0.8)
  α = zero(T)
  αmin = eps(T)^(0.9)

  # Compute RHS g = -(A * q + b)
  mul!(ψ.g, ψ.A, q, -one(T), zero(T))
  ψ.g .-= ψ.b

  # Retrieve qrm workspace
  shifted_spmat = ψ.shifted_spmat
  spmat = shifted_spmat.spmat
  spfct = ψ.spfct
  qrm_update_shift_spmat!(shifted_spmat, α)
  spmat.val[1:spmat.mat.nz - spmat.mat.m] .= ψ.A.vals
  qrm_spfct_init!(spfct, spmat)
  qrm_set(spfct, "qrm_keeph", 0) # Discard de Q matrix in all subsequent QR factorizations
  qrm_set(spfct, "qrm_rd_eps", eps(T)^(0.4)) # If a diagonal element of the R-factor is less than eps(R)^(0.4), we consider that A is rank defficient.

  # Check interior convergence
  qrm_analyse!(spmat, spfct; transp='t')
  _obj_dot_grad!(spmat, spfct, ψ.p, ψ.q, ψ.g, ψ.dq)
  
  # Check full-rankness
  full_row_rank = (qrm_get(spfct,"qrm_rd_num") == 0)
  if !full_row_rank
    # QRMumps cannot factorize rank-deficient matrices; use the Golub-Riley iteration instead
    α = αmin
    qrm_golub_riley!(ψ.shifted_spmat, spfct, ψ.p, ψ.g, ψ.dp, ψ.q, ψ.dq, transp = 't', α = α, tol = eps(T)^(0.75)) # Now, ψ.p = Aᵀψ.q and ψ.q = (AAᵀ)†b

    # Compute residual
    qrm_spmat_mv!(spmat, T(1), ψ.q, T(0), ψ.dp, transp = 't')
    qrm_spmat_mv!(spmat, T(1), ψ.dp, T(0), ψ.dq, transp = 'n')
    @. ψ.dq = ψ.dq - ψ.g # In this case, ψ.dq = AAᵀψ.q - b. If ‖ψ.dq‖₂ is small, then g ∈ Range(AAᵀ)

    if abs(norm(ψ.q) - ν*ψ.h.lambda) < atol && norm(ψ.dq) ≤ eps(T)^(0.5) # Check interior optimality and range of AAᵀ
      y .= ψ.p[1:length(y)]
      y .+= q
      return y 
    end

    # The solution is not α = 0, prepare root finding
    qrm_update_shift_spmat!(shifted_spmat, α)
    _obj_dot_grad!(spmat, spfct, ψ.p, ψ.q, ψ.g, ψ.dq)
  end
  
  # Scalar Root finding
  k = 0
  elapsed_time = time() - start_time
  α₊ = α 

  norm_q = norm(ψ.q)
  if abs(norm_q - ν*ψ.h.lambda) > atol
    while abs(norm_q - ν*ψ.h.lambda) > atol && k < max_iter && elapsed_time < max_time

      α₊ += (norm_q / (ν * ψ.h.lambda) - 1) * (norm_q / norm(ψ.p))^2
      α = α₊ > 0 ? α₊ : θ*α
      α = α ≤ αmin ? αmin : α
      
      qrm_update_shift_spmat!(shifted_spmat, α)

      _obj_dot_grad!(spmat, spfct, ψ.p, ψ.q, ψ.g, ψ.dq)
      norm_q = norm(ψ.q)

      α == αmin && break
      
      k += 1
      elapsed_time = time() - start_time
    end
  end

  k > max_iter && @warn "ShiftedCompositeNormL2: Newton method did not converge during prox computation returning with residual $(abs(norm(ψ.q) - ν*ψ.h.lambda)) instead"
  mul!(y, ψ.A', ψ.q)
  y .+= q
  return y
end

# Utility function that computes in place both q = s(α) and p such that ‖p‖² = s(α)ᵀ∇s(α) for the secular equation.
function _obj_dot_grad!(spmat :: qrm_spmat{T}, spfct :: qrm_spfct{T}, p :: AbstractVector{T}, q :: AbstractVector{T}, g :: AbstractVector{T}, dq :: AbstractVector{T}) where T
  qrm_factorize!(spmat, spfct, transp='t')
  qrm_solve!(spfct, g, p, transp='t')
  qrm_solve!(spfct, p, q, transp='n')
  qrm_refine!(spmat, spfct, q, g, dq, p)
  qrm_solve!(spfct, q, p, transp='t')
end