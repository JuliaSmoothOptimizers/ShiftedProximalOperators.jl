module ShiftedProximalOperators

using ProximalOperators

export PROXPATH
PROXPATH = dirname(@__DIR__)

abstract type ShiftedProximalFunction <: ProximableFunction end

# prox files 
include("shiftedl0norm.jl")
include("ShiftedNormL0BInf.jl")

export prox, shift!

end # module
