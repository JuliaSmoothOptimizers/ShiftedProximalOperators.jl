export ShiftedNormL0

mutable struct ShiftedNormL0 <: ShiftedProximableFunction
	h::ProximableFunction
	x
	λ
	function ShiftedNormL0(n, λ)
		new(NormL0(λ), Vector{Float64}(undef, n), λ)
	end
end

shifted(h::NormL0, x::Vector{Float64}, λ)= ShiftedNormL0(h, x, λ)

function shift!(ψ::ShiftedNormL0, x::Vector{Float64})
	ψ.x .= x
	ψ
end

function prox(ψ::ShiftedNormL0, q, σ) 
 c = sqrt(2 * ψ.λ * σ)
  for i ∈ eachindex(q)
    xpq = ψ.x[i] + q[i]
    absxpq = abs(xpq)
    if absxpq ≤ c
      ψ.s[i] = 0
    else
      ψ.s[i] = xpq
    end
  end
  ψ.s .-= ψ.x
  return ψ.s
end
