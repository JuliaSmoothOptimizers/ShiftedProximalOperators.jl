# Extensions of LinearAlgebra's svd functionality
# with U, S and VT preallocated.
#
# D. Orban, based on Julia's stdlib
# 6/19/22

# Example use:
# julia> A = rand(8, 10)
# julia> F = psvd_workspace(A, full=false, alg=LinearAlgebra.QRIteration())  # preallocates
# julia> B = copy(A);  # optional; the next line destroys B
# julia> psvd!(F, B, full=false, alg=LinearAlgebra.QRIteration())  # does not allocate
# julia> julia> norm(F.U * Diagonal(F.S) * F.Vt - A)
# 2.1215822150677606e-14

# Improvements:
# * use job = 'O' to store U or Vt in A?
# * safeguard user from calling psvd_workspace() and psvd!() with different arguments

import Base.require_one_based_indexing
import LinearAlgebra.BLAS.@blasfunc
import LinearAlgebra.BlasInt, LinearAlgebra.BlasFloat
import LinearAlgebra.Algorithm
import LinearAlgebra.chkstride1
import LinearAlgebra.default_svd_alg
import LinearAlgebra.Factorization
import LinearAlgebra.LAPACK.chklapackerror

export psvd_workspace_qr, psvd_workspace_dd, psvd_qr!, psvd_dd!, psvd

mutable struct PSVD{T, Tr, M <: AbstractArray{T}} <: Factorization{T}
  U::M
  S::Vector{Tr}
  Vt::M
  work::Vector{T}
  iwork::Vector{BlasInt}
  rwork::Vector{Tr}
  function PSVD{T, Tr, M}(U, S, Vt, work, iwork, rwork) where {T, Tr, M <: AbstractArray{T}}
    require_one_based_indexing(U, S, Vt)
    new{T, Tr, M}(U, S, Vt, work, iwork, rwork)
  end
end
PSVD(
  U::AbstractArray{T},
  S::Vector{Tr},
  Vt::AbstractArray{T},
  work::Vector{T},
  iwork::Vector{BlasInt},
  rwork::Vector{Tr},
) where {T, Tr} = PSVD{T, Tr, typeof(U)}(U, S, Vt, work, iwork, rwork)
function PSVD{T}(
  U::AbstractArray,
  S::AbstractVector{Tr},
  Vt::AbstractArray,
  work::AbstractVector{T},
  iwork::AbstractVector{I},
  rwork::AbstractVector{Tr},
) where {T, Tr, I <: Integer}
  PSVD(
    convert(AbstractArray{T}, U),
    convert(Vector{Tr}, S),
    convert(AbstractArray{T}, Vt),
    convert(Vector{T}, work),
    convert(Vector{BlasInt}, iwork),
    convert(Vector{Tr}, rwork),
  )
end

PSVD{T}(F::PSVD) where {T} = PSVD(
  convert(AbstractMatrix{T}, F.U),
  convert(AbstractVector{real(T)}, F.S),
  convert(AbstractMatrix{T}, F.Vt),
  convert(AbstractVector{T}, F.work),
  convert(AbstractVector{BlasInt}, F.iwork),
  convert(AbstractVector{Tr}, F.rwork),
)
Factorization{T}(F::PSVD) where {T} = PSVD{T}(F)

# iteration for destructuring into components
Base.iterate(S::PSVD) = (S.U, Val(:S))
Base.iterate(S::PSVD, ::Val{:S}) = (S.S, Val(:V))
Base.iterate(S::PSVD, ::Val{:V}) = (S.V, Val(:done))
Base.iterate(S::PSVD, ::Val{:done}) = nothing

# Functions for alg = QRIteration()

for (gesvd, elty, relty) in ((:dgesvd_, :Float64, :Float64), (:sgesvd_, :Float32, :Float32))
  @eval begin
    function psvd_workspace_qr(A::StridedMatrix{$elty}; full::Bool = false)
      jobuvt = full ? 'A' : 'S'
      m, n = size(A)
      minmn = min(m, n)
      S = similar(A, $relty, minmn)
      U = similar(A, $elty, jobuvt == 'A' ? (m, m) : (m, minmn))
      Vt = similar(A, $elty, jobuvt == 'A' ? (n, n) : (minmn, n))
      work = Vector{$elty}(undef, 1)
      lwork = BlasInt(-1)
      info = Ref{BlasInt}()
      rwork = Vector{$relty}(undef, 0)
      ccall(
        (@blasfunc($gesvd), libblastrampoline),
        Cvoid,
        (
          Ref{UInt8},
          Ref{UInt8},
          Ref{BlasInt},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{BlasInt},
          Clong,
          Clong,
        ),
        jobuvt,
        jobuvt,
        m,
        n,
        A,
        max(1, stride(A, 2)),
        S,
        U,
        max(1, stride(U, 2)),
        Vt,
        max(1, stride(Vt, 2)),
        work,
        lwork,
        info,
        1,
        1,
      )
      chklapackerror(info[])
      lwork = BlasInt(real(work[1]))
      resize!(work, lwork)
      iwork = Vector{BlasInt}(undef, 0)
      return PSVD(U, S, Vt, work, iwork, rwork)
    end

    # !!! this call destroys the contents of A
    function psvd_qr!(
      F::PSVD{$elty, $relty, M},
      A::StridedMatrix{$elty};
      full::Bool = false,
    ) where {M}
      jobuvt = full ? 'A' : 'S'
      m, n = size(A)
      m, n = size(A)
      minmn = min(m, n)
      @assert length(F.S) == minmn
      @assert size(F.U) == (jobuvt == 'A' ? (m, m) : (m, minmn))
      @assert size(F.Vt) == (jobuvt == 'A' ? (n, n) : (minmn, n))
      lwork = length(F.work)
      info = Ref{BlasInt}()
      ccall(
        (@blasfunc($gesvd), libblastrampoline),
        Cvoid,
        (
          Ref{UInt8},
          Ref{UInt8},
          Ref{BlasInt},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{BlasInt},
          Clong,
          Clong,
        ),
        jobuvt,
        jobuvt,
        m,
        n,
        A,
        max(1, stride(A, 2)),
        F.S,
        F.U,
        max(1, stride(F.U, 2)),
        F.Vt,
        max(1, stride(F.Vt, 2)),
        F.work,
        lwork,
        info,
        1,
        1,
      )
      chklapackerror(info[])
      return F
    end
  end
end

for (gesvd, elty, relty) in ((:zgesvd_, :ComplexF64, :Float64), (:cgesvd_, :ComplexF32, :Float32))
  @eval begin
    function psvd_workspace_qr(A::StridedMatrix{$elty}; full::Bool = false)
      jobuvt = full ? 'A' : 'S'
      minmn = min(m, n)
      S = similar(A, $relty, minmn)
      U = similar(A, $elty, jobuvt == 'A' ? (m, m) : (m, minmn))
      Vt = similar(A, $elty, jobuvt == 'A' ? (n, n) : (minmn, n))
      work = Vector{$elty}(undef, 1)
      lwork = BlasInt(-1)
      info = Ref{BlasInt}()
      rwork = Vector{R}(undef, 5minmn)
      ccall(
        (@blasfunc($gesvd), libblastrampoline),
        Cvoid,
        (
          Ref{UInt8},
          Ref{UInt8},
          Ref{BlasInt},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$relty},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$relty},
          Ptr{BlasInt},
          Clong,
          Clong,
        ),
        jobu,
        jobvt,
        m,
        n,
        A,
        max(1, stride(A, 2)),
        S,
        U,
        max(1, stride(U, 2)),
        Vt,
        max(1, stride(Vt, 2)),
        work,
        lwork,
        rwork,
        info,
        1,
        1,
      )
      chklapackerror(info[])
      lwork = BlasInt(real(work[1]))
      resize!(work, lwork)
      iwork = Vector{BlasInt}(undef, 0)
      return PSVD(U, S, Vt, work, iwork, rwork)
    end

    # !!! this call destroys the contents of A
    function psvd_qr!(
      F::PSVD{$elty, $relty, M},
      A::StridedMatrix{$elty};
      full::Bool = false,
    ) where {M}
      jobuvt = full ? 'A' : 'S'
      m, n = size(A)
      minmn = min(m, n)
      @assert length(F.S) == minmn
      @assert size(F.U) == (jobuvt == 'A' ? (m, m) : (m, minmn))
      @assert size(F.Vt) == (jobuvt == 'A' ? (n, n) : (minmn, n))
      lwork = length(F.work)
      info = Ref{BlasInt}()
      ccall(
        (@blasfunc($gesvd), libblastrampoline),
        Cvoid,
        (
          Ref{UInt8},
          Ref{UInt8},
          Ref{BlasInt},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$relty},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$relty},
          Ptr{BlasInt},
          Clong,
          Clong,
        ),
        jobu,
        jobvt,
        m,
        n,
        A,
        max(1, stride(A, 2)),
        F.S,
        F.U,
        max(1, stride(F.U, 2)),
        F.Vt,
        max(1, stride(F.Vt, 2)),
        F.work,
        lwork,
        F.rwork,
        info,
        1,
        1,
      )
      chklapackerror(info[])
      return F
    end
  end
end

# Functions for alg = DivideAndConquer()

for (gesdd, elty, relty) in ((:dgesdd_, :Float64, :Float64), (:sgesdd_, :Float32, :Float32))
  @eval begin
    function psvd_workspace_dd(A::StridedMatrix{$elty}; full::Bool = false)
      require_one_based_indexing(A)
      chkstride1(A)
      job = full ? 'A' : 'S'
      m, n = size(A)
      minmn = min(m, n)
      U = similar(A, $elty, job == 'A' ? (m, m) : (m, minmn))
      Vt = similar(A, $elty, job == 'A' ? (n, n) : (minmn, n))
      work = Vector{$elty}(undef, 1)
      lwork = BlasInt(-1)
      S = similar(A, $relty, minmn)
      rwork = Vector{$relty}(undef, 0)
      iwork = Vector{BlasInt}(undef, 8 * minmn)
      info = Ref{BlasInt}()
      ccall(
        (@blasfunc($gesdd), libblastrampoline),
        Cvoid,
        (
          Ref{UInt8},
          Ref{BlasInt},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{BlasInt},
          Ptr{BlasInt},
          Clong,
        ),
        job,
        m,
        n,
        A,
        max(1, stride(A, 2)),
        S,
        U,
        max(1, stride(U, 2)),
        Vt,
        max(1, stride(Vt, 2)),
        work,
        lwork,
        iwork,
        info,
        1,
      )
      chklapackerror(info[])
      # Work around issue with truncated Float32 representation of lwork in
      # sgesdd by using nextfloat. See
      # http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=13&t=4587&p=11036&hilit=sgesdd#p11036
      # and
      # https://github.com/scipy/scipy/issues/5401
      lwork = round(BlasInt, nextfloat(real(work[1])))
      resize!(work, lwork)
      return PSVD(U, S, Vt, work, iwork, rwork)
    end

    # !!! this call destroys the contents of A
    function psvd_dd!(
      F::PSVD{$elty, $relty, M},
      A::StridedMatrix{$elty};
      full::Bool = false,
    ) where {M}
      job = full ? 'A' : 'S'
      m, n = size(A)
      minmn = min(m, n)
      @assert length(F.S) == minmn
      @assert size(F.U) == (job == 'A' ? (m, m) : (m, minmn))
      @assert size(F.Vt) == (job == 'A' ? (n, n) : (minmn, n))
      info = Ref{BlasInt}()
      lwork = length(F.work)
      ccall(
        (@blasfunc($gesdd), libblastrampoline),
        Cvoid,
        (
          Ref{UInt8},
          Ref{BlasInt},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{BlasInt},
          Ptr{BlasInt},
          Clong,
        ),
        job,
        m,
        n,
        A,
        max(1, stride(A, 2)),
        F.S,
        F.U,
        max(1, stride(F.U, 2)),
        F.Vt,
        max(1, stride(F.Vt, 2)),
        F.work,
        lwork,
        F.iwork,
        info,
        1,
      )
      chklapackerror(info[])
      return F
    end
  end
end

for (gesdd, elty, relty) in ((:zgesdd_, :ComplexF64, :Float64), (:cgesdd_, :ComplexF32, :Float32))
  @eval begin
    function psvd_workspace_dd(A::StridedMatrix{$elty}; full::Bool = false)
      require_one_based_indexing(A)
      chkstride1(A)
      job = full ? 'A' : 'S'
      m, n = size(A)
      minmn = min(m, n)
      U = similar(A, $elty, job == 'A' ? (m, m) : (m, minmn))
      Vt = similar(A, $elty, job == 'A' ? (n, n) : (minmn, n))
      work = Vector{$elty}(undef, 1)
      lwork = BlasInt(-1)
      S = similar(A, $relty, minmn)
      rwork = Vector{$relty}(undef, minmn * max(5 * minmn + 7, 2 * max(m, n) + 2 * minmn + 1))
      iwork = Vector{BlasInt}(undef, 8 * minmn)
      info = Ref{BlasInt}()
      ccall(
        (@blasfunc($gesdd), libblastrampoline),
        Cvoid,
        (
          Ref{UInt8},
          Ref{BlasInt},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$relty},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$relty},
          Ptr{BlasInt},
          Ptr{BlasInt},
          Clong,
        ),
        job,
        m,
        n,
        A,
        max(1, stride(A, 2)),
        S,
        U,
        max(1, stride(U, 2)),
        Vt,
        max(1, stride(Vt, 2)),
        work,
        lwork,
        rwork,
        iwork,
        info,
        1,
      )
      chklapackerror(info[])
      # Work around issue with truncated Float32 representation of lwork in
      # sgesdd by using nextfloat. See
      # http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=13&t=4587&p=11036&hilit=sgesdd#p11036
      # and
      # https://github.com/scipy/scipy/issues/5401
      lwork = round(BlasInt, nextfloat(real(work[1])))
      resize!(work, lwork)
      rwork = Vector{$relty}(undef, 0)
      return PSVD(U, S, Vt, work, iwork, rwork)
    end

    # !!! this call destroys the contents of A
    function psvd_dd!(
      F::PSVD{$elty, $relty, M},
      A::StridedMatrix{$elty};
      full::Bool = false,
    ) where {M}
      job = full ? 'A' : 'S'
      m, n = size(A)
      minmn = min(m, n)
      @assert length(F.S) == minmn
      @assert size(F.U) == job == 'A' ? (m, m) : (m, minmn)
      @assert size(F.Vt) == job == 'A' ? (n, n) : (minmn, n)
      info = Ref{BlasInt}()
      lwork = length(F.work)
      ccall(
        (@blasfunc($gesdd), libblastrampoline),
        Cvoid,
        (
          Ref{UInt8},
          Ref{BlasInt},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$relty},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$elty},
          Ref{BlasInt},
          Ptr{$relty},
          Ptr{BlasInt},
          Ptr{BlasInt},
          Clong,
        ),
        job,
        m,
        n,
        A,
        max(1, stride(A, 2)),
        S,
        U,
        max(1, stride(U, 2)),
        VT,
        max(1, stride(VT, 2)),
        F.work,
        lwork,
        F.rwork,
        F.iwork,
        info,
        1,
      )
      chklapackerror(info[])
      return F
    end
  end
end

function psvd(
  A::StridedMatrix{T};
  full::Bool = false,
  alg::Algorithm = default_svd_alg(A),
) where {T <: BlasFloat}
  m, n = size(A)
  if m == 0 || n == 0
    u, s, vt = (Matrix{T}(I, m, full ? m : n), real(zeros(T, 0)), Matrix{T}(I, n, n))
    Tr = real(T)
    return PSVD(u, s, vt, T[], BlasInt[], Tr[])
  else
    if typeof(alg) <: LinearAlgebra.QRIteration
      F = psvd_workspace_qr(A, full = full)
      return psvd_qr!(F, copy(A), full = full)
    else
      F = psvd_workspace_dd(A, full = full)
      return psvd_dd!(F, copy(A), full = full)
    end
  end
end
