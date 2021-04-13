module ShiftedProximalOperators

using ProximalOperators, LinearAlgebra

export PROXPATH
PROXPATH = dirname(@__DIR__)

abstract type ShiftedProximalFunction <: ProximableFunction end

export prox, shift!

# prox files 
include("shiftedl0norm.jl")

end # module
