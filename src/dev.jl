
### Des commandes pour tester la lib

#=

h = NormL0(1.0)
n = 10000000
Δ = 2 * rand()
q = 2 * (rand(n) .- 0.5)
χ = NormLinf(1.0)
ν = rand()
xk = rand(n) .- 0.5
ψ = shifted(h, xk, Δ, χ) # idee : shifted(h, xk, l, u, χ)
ShiftedProximalOperators.prox(ψ, q, ν)

=#

# TEST 


include("ShiftedProximalOperators.jl")
#using Plots


h = NormL0(5.0)
n = 10000000
ν = rand()
l = -5.0 ; u = 5*rand(n)
q = 20*(rand(n).-0.5)

# shift once
xk = rand(n) .- 0.5
ψ = shifted(h, xk, l, u)

# check prox
p = prox(ψ, q, ν)

#=

# shift a second time
sj = rand(n) .- 0.5
ω = shifted(ψ, sj)

p = prox(ω, q, ν)


# plot

for i in 1:n
  dom_x = LinRange(ω.l[i], ω.u[i],1000)
  F_x = (dom_x .- q[i]).^2 + 2 * ω.λ * ν .* ((ω.xk[i] + ω.sj[i] .+ dom_x) .!= 0.0)
  x_min = p[i]
  F_min = (x_min - q[i])^2 + 2 * ω.λ * ν * ((ω.xk[i] + ω.sj[i] + x_min) != 0.0)
  if i == 1
    plot(dom_x,F_x,label="F_"*string(i))
  else
    plot!(dom_x,F_x,label="F_"*string(i))
  end
  scatter!([x_min],[F_min],label="min of F_"*string(i))
  #vline!([-ω.xk[i] - ω.sj[i]])
  #hline!([F_min])
end 
current()

=#

#=

h = NormL1(2.0)
n = 5
ν = rand()
l = -10*rand(n)
u = 10*rand(n)
q = 20*(rand(n).-0.5)

# shift once
xk = rand(n) .- 0.5
ψ = shifted(h, xk, l, u)

# check prox
p = prox(ψ, q, ν)

# shift a second time
sj = rand(n) .- 0.5
ω = shifted(ψ, sj)

p = prox(ω, q, ν)



# plot

for i in 1:n
  dom_x = LinRange(ω.l[i], ω.u[i],1000)
  F_x = (dom_x .- q[i]).^2 + 2 * ω.λ * ν .* abs.(ω.xk[i] + ω.sj[i] .+ dom_x)
  x_min = p[i]
  F_min = (x_min - q[i])^2 + 2 * ω.λ * ν * abs.(ω.xk[i] + ω.sj[i] + x_min)
  if i == 1
    plot(dom_x,F_x,label="F_"*string(i))
  else
    plot!(dom_x,F_x,label="F_"*string(i))
  end
  scatter!([x_min],[F_min],label="min of F_"*string(i))
end 
current()

=#