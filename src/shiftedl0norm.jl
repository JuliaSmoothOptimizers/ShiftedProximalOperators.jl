export ShiftedNormL0

mutable struct ShiftedNormL0{R <: Real} <: ProximableFunction
	h
	x
	s
	λ
	function ShiftedNormL0(x::AbstractVector{R}, λ::R)
		new(x, similar(x), λ)
	end
end

(ψ::ShiftedProximalFunction)(y) = ψ.h (ψ.x + y)

shifted(h::NormL0{R}, x::AbstractVector{R}, λ::R) = ShiftedNormL0{R}(h, x, λ)

function shift!(ψ::ShiftedNormL0{R}, x::AbstractVector{R})
	ψ.x .= x
	ψ
end

function prox(ψ::ShiftedNormL0, q, σ)
	c = sqrt(2 * σ)
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
