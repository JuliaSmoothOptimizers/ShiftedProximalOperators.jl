module ShiftedProximalOperators

using ProximalOperators

abstract type ShiftedProximableFunction <: ProximableFunction end

# prox files 
include("shiftedl0norm.jl")
include("ShiftedNormL0BInf.jl")

(ψ::ShiftedProximableFunction)(y) = ψ.h(ψ.x + y)

export prox, shift!

end # module
