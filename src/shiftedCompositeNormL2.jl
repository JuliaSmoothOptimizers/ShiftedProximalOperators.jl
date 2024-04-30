export ShiftedCompositeNormL2

@doc raw"""
    ShiftedCompositeNormL2(h, c!, J!, A, b)

Returns the shift of a function c composed with the ``\ell_{2}`` norm (see CompositeNormL2.jl).
Here, c is linearized i.e, ``c(x+s) \approx c(x) + J(x)s``. 
```math
f(s) = λ \|c(x) + J(x)s\|_2
```
where ``\lambda > 0``. c! and J! should be functions
```math
\begin{aligned}
&c(x) : \mathbb{R}^n \xrightarrow[]{} \mathbb{R}^m \\
&J(x) : \mathbb{R}^n \xrightarrow[]{} \mathbb{R}^{m\times n}
\end{aligned}
```
such that J is the Jacobian of c. A and b should respectively be a matrix and a vector which can respectively store the values of J and c.
A is expected to be sparse, c and J should have signatures
c!(b <: AbstractVector{Real}, xk <: AbstractVector{Real})
J!(A <: AbstractSparseMatrixCOO{Real,Integer}, xk <: AbstractVector{Real})
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
  Aᵧ::V2
  b::V3
  g::V3
  res::V3
  sol::V3
  dsol::V3
  p::V3
  dp
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
    Aᵧ = SparseMatrixCOO(A.m,A.n + A.m, similar(A.rows,length(A.rows) + A.m), similar(A.cols, length(A.cols) + A.m), similar(A.vals,length(A.vals) + A.m))
    if length(b) != size(A,1)
      error("ShiftedCompositeNormL2: Wrong input dimensions, there should be as many constraints as rows in the Jacobian")
    end
    new{R,typeof(c!),typeof(J!),typeof(A),typeof(b)}(NormL2(λ),c!,J!,A,Aᵧ,b,g,res,sol,dsol,p,dp)
  end
end


shifted(λ::R, c!::Function, J!::Function, A::AbstractMatrix{R}, b::AbstractVector{R}, xk :: AbstractVector{R}) where {R <: Real} = begin
  c!(b,xk)
  J!(A,xk)
  ShiftedCompositeNormL2(λ,c!,J!,A,b)
end

shifted(
  ψ::ShiftedCompositeNormL2{R, V0, V1, V2, V3},
  xk::AbstractVector{R},
) where {R <: Real, V0 <: Function, V1 <: Function, V2 <: AbstractMatrix{R}, V3<: AbstractVector{R}} = begin
  b = similar(ψ.b)
  ψ.c!(b,xk)
  A = similar(ψ.A)
  ψ.J!(A,xk)
  ShiftedCompositeNormL2(ψ.h.lambda, ψ.c!, ψ.J!, A, b)
end
 
shifted(
  ψ::CompositeNormL2{R, V0, V1, V2, V3},
  xk::AbstractVector{R}
) where {R <: Real, V0 <: Function, V1 <: Function, V2 <: AbstractMatrix{R}, V3 <: AbstractVector{R}} = begin
  b = similar(ψ.b)
  ψ.c!(b,xk)
  A = copy(ψ.A)
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
  maxiter = 10000
) where {R <: Real, V0 <: Function,V1 <:Function,V2 <: AbstractMatrix{R}, V3 <: AbstractVector{R}}

  # Initialize Aᵧ
  ψ.Aᵧ.rows .= [ψ.A.rows;collect(eltype(ψ.A.rows),1:ψ.A.m)] 
  ψ.Aᵧ.cols .= [ψ.A.cols;collect(eltype(ψ.A.cols),ψ.A.n+1:ψ.A.n + ψ.A.m)]
  ψ.Aᵧ.vals .= [ψ.A.vals;zeros(eltype(ψ.Aᵧ.vals),ψ.A.m)]

  mul!(ψ.g, ψ.A, q)
  ψ.g .+= ψ.b

  spmat = qrm_spmat_init(ψ.Aᵧ; sym=false)
  spfct = qrm_spfct_init(spmat)
  qrm_analyse!(spmat, spfct; transp='t')
  qrm_set(spfct, "qrm_keeph", 0)
  qrm_factorize!(spmat, spfct, transp='t')

  qrm_solve!(spfct, ψ.g, ψ.p, transp='t')
  qrm_solve!(spfct, ψ.p, ψ.sol, transp='n')

  # 1 Step of iterative refinement
  ψ.res .= ψ.g

  mul!(ψ.dp, ψ.Aᵧ', ψ.sol)
  mul!(ψ.dsol, ψ.Aᵧ, ψ.dp)

  ψ.res .-= ψ.dsol
  if norm(ψ.res) > eps(R)^0.75
    qrm_solve!(spfct, ψ.res, ψ.dp, transp='t')
    qrm_solve!(spfct, ψ.dp, ψ.dsol, transp='n')
    ψ.sol .+= ψ.dsol
    ψ.p .+= ψ.dp
  end  

  ψ.sol .*= -1

  # Scalar Root finding
  γ = 0.0
  k = 0
  if norm(ψ.sol) > σ*ψ.h.lambda

    # Do we need step of iterative refinement for this one ?
    qrm_solve!(spfct, ψ.sol, ψ.p, transp='t')

    while abs(norm(ψ.sol) - σ*ψ.h.lambda) > eps(R)^0.75 && k < maxiter

      γ += (norm(ψ.sol)/(σ*ψ.h.lambda) - 1.0)*(norm(ψ.sol)/norm(ψ.p))^2
      
      qrm_update!(spmat,[ψ.A.vals; fill(eltype(ψ.A.vals)(sqrt(γ)),ψ.A.m)])
      qrm_factorize!(spmat,spfct, transp='t')

      qrm_solve!(spfct, ψ.g, ψ.p, transp='t')
      qrm_solve!(spfct, ψ.p, ψ.sol, transp='n')
      qrm_solve!(spfct, ψ.sol, ψ.p, transp='t')

      # 1 Step of iterative refinement
      ψ.res .= ψ.g

      mul!(ψ.dp, ψ.Aᵧ', ψ.sol)
      mul!(ψ.dsol, ψ.Aᵧ, ψ.dp)

      ψ.res .-= ψ.dsol
      if norm(ψ.res) > eps(R)^0.75
        qrm_solve!(spfct, ψ.res, ψ.dp, transp='t')
        qrm_solve!(spfct, ψ.dp, ψ.dsol, transp='n')
        qrm_solve!(spfct, ψ.dsol, ψ.dp, transp='t')
        ψ.sol .+= ψ.dsol
        ψ.p .+= ψ.dp
      end  
      
      ψ.sol .*= -1
      k += 1
    end
  end

  k < maxiter || @warn "ShiftedCompositeNormL2: Newton method did not converge during prox computation returning y with residue $(abs(norm(ψ.sol) - σ*ψ.h.lambda)) instead"
  mul!(y, ψ.A', ψ.sol)
  y .+= q
  return y
end