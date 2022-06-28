@testset "psvd" begin
  A = rand(8, 10)
  # test simple mode
  F = psvd(A, full = false)
  @test norm(F.U * Diagonal(F.S) * F.Vt - A) / norm(A) ≤ 100 * eps(eltype(A))

  # test expert mode
  F = psvd_workspace_qr(A, full = false)
  B = copy(A)
  psvd_qr!(F, B, full = false)
  @test norm(F.U * Diagonal(F.S) * F.Vt - A) / norm(A) ≤ 100 * eps(eltype(A))

  B = copy(A)
  @test (@allocated psvd_qr!(F, B, full = false)) == 0
end
