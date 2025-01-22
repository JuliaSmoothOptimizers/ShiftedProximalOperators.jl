export ShiftedCompositeNormL2
@doc raw"""
    ShiftedCompositeNormL2(h, c!, J!, A, b)

Returns the shift of a function ``c`` composed with the ``ℓ₂`` norm (see CompositeNormL2.jl).
Here, ``c`` is linearized i.e, ``c(x+s) ≈ c(x) + J(x)s``. 
```math
f(s) = λ ‖c(x) + J(x)s‖₂,
```
where ``λ > 0``. `c!` and `J!` should implement functions 
```math
c : ℜⁿ ↦ ℜᵐ,
```
```math
J : ℜⁿ ↦ ℜᵐˣⁿ,   
```
such that ``J`` is the Jacobian of ``c``. `A` and `b` should respectively be a matrix and a vector which can respectively store the values of ``J`` and ``c``.
`A` is expected to be sparse, `c!` and `J!` should have signatures
```
c!(b <: AbstractVector{Real}, xk <: AbstractVector{Real})
J!(A <: AbstractSparseMatrixCOO{Real,Integer}, xk <: AbstractVector{Real})
```
"""
mutable struct ShiftedCompositeNormL2{
  R <: Real,
  V0 <: Function,
  V1 <: Function,
  V2 <: AbstractMatrix{R},
  V3 <: AbstractVector{R},
} <: ShiftedCompositeProximableFunction
  h::NormL2{R}
  c!::V0
  J!::V1
  A::V2
  Aα::V2
  spmat::qrm_spmat{R}
  spfct::qrm_spfct{R}
  b::V3
  g::V3
  res::V3
  sol::V3
  dsol::V3
  p::V3
  dp::V3
  function ShiftedCompositeNormL2(
    λ::R,
    c!::Function,
    J!::Function,
    A::AbstractMatrix{R},
    b::AbstractVector{R},
  ) where {R <: Real}
    p = similar(b, A.n + A.m)
    dp = similar(b, A.n + A.m)
    g = similar(b)
    res = similar(b)
    sol = similar(b)
    dsol = similar(b)
    Aα = SparseMatrixCOO(A.m,A.n + A.m, similar(A.rows,length(A.rows) + A.m), similar(A.cols, length(A.cols) + A.m), similar(A.vals,length(A.vals) + A.m))
    if length(b) != size(A,1)
      error("ShiftedCompositeNormL2: Wrong input dimensions, there should be as many constraints as rows in the Jacobian")
    end
    Aα.rows[1:length(A.rows)] .= A.rows
    Aα.rows[length(A.rows)+1:end] .= eltype(A.rows)(1):eltype(A.rows)(A.m)
    Aα.cols[1:length(A.cols)] .= A.cols
    Aα.cols[length(A.cols)+1:end] .= eltype(A.cols)(A.n+1):eltype(A.cols)(A.n + A.m)

    spmat = qrm_spmat_init(Aα; sym=false)
    spfct = qrm_spfct_init(spmat)
    qrm_set(spfct, "qrm_keeph", 0) # Discard de Q matrix in all subsequent QR factorizations
    qrm_set(spfct, "qrm_rd_eps", eps(R)^(0.4)) # If a diagonal elemnt of the R-factor is less than eps(R)^(0.4), we consider that A is rank defficient.
    new{R,typeof(c!),typeof(J!),typeof(A),typeof(b)}(NormL2(λ), c!, J!, A, Aα, spmat, spfct, b, g, res, sol, dsol, p, dp)
  end
end
 
shifted(
  ψ::CompositeNormL2{R, V0, V1, V2, V3},
  xk::AbstractVector{R}
) where {R <: Real, V0 <: Function, V1 <: Function, V2 <: AbstractMatrix{R}, V3 <: AbstractVector{R}} = begin
  b = similar(ψ.b)
  ψ.c!(b,xk)
  A = similar(ψ.A)
  ψ.J!(A,xk)
  ShiftedCompositeNormL2(ψ.h.lambda, ψ.c!, ψ.J!, A, b)
end

fun_name(ψ::ShiftedCompositeNormL2) = "shifted L2 norm"
fun_expr(ψ::ShiftedCompositeNormL2) = "t ↦ ‖c(xk) + J(xk)t‖₂"
fun_params(ψ::ShiftedCompositeNormL2) = "c(xk) = $(ψ.b)\n" * " "^14 * "J(xk) = $(ψ.A)\n"

function prox!(
  y::AbstractVector{R},
  ψ::ShiftedCompositeNormL2{R, V0, V1, V2, V3},
  q::AbstractVector{R},
  σ::R;
  max_iter = 10000,
  max_time = 180.0
) where {R <: Real, V0 <: Function,V1 <:Function,V2 <: AbstractMatrix{R}, V3 <: AbstractVector{R}}

  start_time = time()
  θ = R(0.8)
  α = R(0.0)
  αmin = eps(R)^(0.75)

  # Initialize Aα
  ψ.Aα.vals[1:length(ψ.A.vals)] .= ψ.A.vals
  ψ.Aα.vals[length(ψ.A.vals)+1:end] .= eltype(ψ.A.vals)(0)

  mul!(ψ.g, ψ.A, q)
  ψ.g .+= ψ.b
  ψ.g .*= -1

  spmat = ψ.spmat
  spfct = ψ.spfct

  qrm_update!(spmat, ψ.Aα.vals)
  # Check interior convergence
  qrm_analyse!(spmat, spfct; transp='t')
  qrm_factorize!(spmat, spfct, transp='t')

  qrm_solve!(spfct, ψ.g, ψ.p, transp='t')
  qrm_solve!(spfct, ψ.p, ψ.sol, transp='n')
  qrm_solve!(spfct, ψ.sol, ψ.p, transp='t')
  _iterative_refinement!(spfct,ψ)
  
  full_row_rank = !(qrm_get(spfct,"qrm_rd_num") > 0)
  if !full_row_rank
    #TODO: once Golub-Riley has been implemented in QRMumps, this should be replaced with a call to qrm_golub_riley
    α = αmin
    
    ψ.Aα.vals[length(ψ.A.vals)+1:end] .= eltype(ψ.A.vals)(sqrt(α))
    qrm_update!(spmat, ψ.Aα.vals)
    qrm_factorize!(spmat, spfct, transp='t')

    qrm_solve!(spfct, ψ.g, ψ.p, transp='t')
    qrm_solve!(spfct, ψ.p, ψ.sol, transp='n')
    qrm_solve!(spfct, ψ.sol, ψ.p, transp='t')
    _iterative_refinement!(spfct, ψ)

    α₊ = α + (norm(ψ.sol)/(σ*ψ.h.lambda) - 1.0)*(norm(ψ.sol)/norm(ψ.p))^2
    if norm(ψ.sol) ≤ σ*ψ.h.lambda + eps(R) && α₊ ≤ α    # We consider ψ.sol is a good approximation of the least-norm solution
      mul!(y, ψ.A', ψ.sol)
      y .+= q
      return y
    end
  end
  
  # Scalar Root finding
  k = 0
  elapsed_time = time() - start_time
  α₊ = α 
  if norm(ψ.sol) > σ*ψ.h.lambda || !full_row_rank
    while norm(ψ.sol) > σ*ψ.h.lambda + eps(R)^0.75 && k < max_iter && elapsed_time < max_time

      solNorm = norm(ψ.sol)
      α₊ += (solNorm / (σ * ψ.h.lambda) - 1) * (solNorm / norm(ψ.p))^2
      α = α₊ > 0 ? α₊ : θ*α
      α = α ≤ αmin ? αmin : α
      
      ψ.Aα.vals[length(ψ.A.vals)+1:end] .= eltype(ψ.A.vals)(sqrt(α))
      qrm_update!(spmat, ψ.Aα.vals)
      qrm_factorize!(spmat,spfct, transp='t')

      qrm_solve!(spfct, ψ.g, ψ.p, transp='t')
      qrm_solve!(spfct, ψ.p, ψ.sol, transp='n')
      qrm_solve!(spfct, ψ.sol, ψ.p, transp='t')
      _iterative_refinement!(spfct, ψ)

      α == αmin && break
      
      k += 1
      elapsed_time = time() - start_time
    end
  end

  #Sometimes alpha tends to 0, we don't to print the residual in this case, it is usually huge.
  (k > max_iter && α > eps(R)) && @warn "ShiftedCompositeNormL2: Newton method did not converge during prox computation returning with residue $(abs(norm(ψ.sol) - σ*ψ.h.lambda)) instead"
  mul!(y, ψ.A', ψ.sol)
  y .+= q
  return y
end

#TODO: remove this and use qrm_iterative_refinement 
function _iterative_refinement!(
  spfct, 
  ψ::ShiftedCompositeNormL2{R, V0, V1, V2, V3}; 
  atol = sqrt(eps(R)), 
  rtol = sqrt(eps(R))
  ) where{R, V0, V1, V2, V3}

  ψ.res .= ψ.g

  mul!(ψ.dp, ψ.Aα', ψ.sol)
  mul!(ψ.dsol, ψ.Aα, ψ.dp)

  ψ.res .-= ψ.dsol
  if norm(ψ.res) > atol + rtol*norm(ψ.g)
    qrm_solve!(spfct, ψ.res, ψ.dp, transp='t')
    qrm_solve!(spfct, ψ.dp, ψ.dsol, transp='n')
    qrm_solve!(spfct, ψ.dsol, ψ.dp, transp='t')
    ψ.sol .+= ψ.dsol
    ψ.p .+= ψ.dp
  end
end