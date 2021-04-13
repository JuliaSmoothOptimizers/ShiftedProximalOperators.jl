export ShiftedNormL0BInf

mutable struct ShiftedNormL0BInf <: ShiftedProximableFunction
	h::ProximableFunction
	x
  s
	λ
	Δ
	function ShiftedNormL0BInf(n, λ)
		new(NormL0(λ), Vector{Float64}(undef, n), Vector{Float64}(undef, n), λ, 0.0)
	end
end

shifted(h::NormL0, x::Vector{Float64},s::Vector{Float64}(undef, n), λ, Δ)= ShiftedNormL0BInf(h, x, s, λ, Δ)

function shift!(ψ::ShiftedNormL0BInf, x::Vector{Float64}, Δ)
	ψ.Δ = Δ
	ψ.x .= x
	ψ
end

function prox(ψ::ShiftedNormL0BInf,  q, σ)
	ProjB!(y) = begin
		for i ∈ eachindex(y)
			y[i] = min(max(y[i], ψ.x[i] - ψ.Δ), ψ.x[i] +ψ.Δ)
		end
	end 
	# @show σ/λ, λ
	c = sqrt(2*ψ.λ*σ)
	w = ψ.x+q

	for i = 1:length(w)
		absx = abs(w[i])
		if absx <=c
			ψ.s[i] = 0
		else
			ψ.s[i] = w[i]
		end
	end
	ProjB!(ψ.s) 
  ψ.s .-= ψ.x
	return ψ.s 
end
