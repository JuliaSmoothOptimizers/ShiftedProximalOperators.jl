module ShiftedProximalOperators

using ProximalOperators


export prox, shift!, ShiftedProximableFunction

abstract type ShiftedProximableFunction <: ProximableFunction end



# prox files 
include("shiftedl0norm.jl")
include("ShiftedNormL0BInf.jl")

(ψ::ShiftedProximableFunction)(y) = ψ.h(ψ.x + y)



end # module
