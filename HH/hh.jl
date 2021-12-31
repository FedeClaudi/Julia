using StaticArrays: SA, @MVector
using Plots
using DifferentialEquations: ODEProblem, solve, Tsit5

gr()

"""
    Julia implementation of the Hodgkin-Huxley model of single neuron membrane dynamics.
    Adapted from a tutorial in Python:
    https://hodgkin-huxley-tutorial.readthedocs.io/en/latest/_static/Hodgkin%20Huxley.html
"""


# -------------------------- channel gating kinetics ------------------------- #
alphaM(V::Float64) = 0.1*(V+40.0)/(1.0 - exp(-(V+40.0) / 10.0))

betaM(V::Float64) = 4.0*exp(-(V+65.0) / 18.0)

alphaH(V::Float64) = 0.07*exp(-(V+65.0) / 20.0)

betaH(V::Float64) = 1.0/(1.0 + exp(-(V+35.0) / 10.0))

alphaN(V::Float64) = 0.01*(V+55.0)/(1.0 - exp(-(V+55.0) / 10.0))

betaN(V::Float64) = 0.125*exp(-(V+65) / 80.0)


alphaM(V) = 0.1*(V+40.0)/(1.0 - exp(-(V+40.0) / 10.0))

betaM(V) = 4.0*exp(-(V+65.0) / 18.0)

alphaH(V) = 0.07*exp(-(V+65.0) / 20.0)

betaH(V) = 1.0/(1.0 + exp(-(V+35.0) / 10.0))

alphaN(V) = 0.01*(V+55.0)/(1.0 - exp(-(V+55.0) / 10.0))

betaN(V) = 0.125*exp(-(V+65) / 80.0)

# ---------------------------- currents functions ---------------------------- #
I_Na(V::Float64, m::Float64, h::Float64, gNa::Float64, ENa::Float64) = gNa * m^3 * h * (V - ENa)
I_K(V::Float64, n::Float64, gK::Float64, EK::Float64) = gK * n^4 * (V-EK)
I_L(V::Float64, gL::Float64, EL::Float64) = gL * (V - EL)

I_Na(V, m, h, gNa, ENa) = gNa * m^3 * h * (V - ENa)
I_K(V, n, gK, EK) = gK * n^4 * (V-EK)
I_L(V, gL, EL) = gL * (V - EL)

# --------------------------------- HH model --------------------------------- #
"""
    Time varying input corrent
"""
function I₀(t) 
    # if t < 100
    #     return 0.0
    # elseif t < 200
    #     return 6.25
    # elseif t < 300
    #     return 0.0
    # elseif t < 400
    #     return 20.0
    # else
    #     return 0.0
    # end
    return 40.0
end

"""
    Differential equations for HH model dynamics
"""
function hh_dynamics(x, p, t)
    # state vars
    V₀, m₀, h₀, n₀ = x

    # params
    Cm, gNa, gK, gL, ENa, EK, EL = p

    V₁ = (I₀(t) - I_Na(V₀, m₀, h₀, gNa, ENa) - I_K(V₀, n₀, gK, EK) - I_L(V₀, gL, EL))/Cm
    
    m₁ = alphaM(V₀)*(1-m₀) - betaM(V₀)*m₀
    h₁ = alphaH(V₀)*(1-h₀) - betaH(V₀)*h₀
    n₁ = alphaN(V₀)*(1-n₀) - betaN(V₀)*n₀
    # SA[V₁, m₁, h₁, n₁] 
    @MVector [V₁, m₁, h₁, n₁] 
end

function HH(;duration::Float64=1.0, dt::Float64=0.01)
    # define an initial state and a params array
    state = SA[
        -65.0,  # [V] initial voltage
        0.05,  # m
        0.6,  # h
        0.32,  # n
    ]
    params = SA[
        # capacitance
        1.0,    # [μF/cm²] - Cm
        # max conductances
        120,    # [mS/cm²] - gNa
        36,     # [mS/cm²] - gK
        0.3,    # [mS/cm²] - gL
        # Nerst potentials
        50,     # [mV] - ENa
        -77,    # [mV] - EK
        -54.387  # [mV] - EL
    ]
 
    # solve problem
    tspan = (0.0, duration)
    prob = ODEProblem(hh_dynamics, state, tspan, params)
    sol = solve(prob, Tsit5(); saveat=dt)

    return sol
end


function plot_results(sol)
    @info "Showing plots"

    V = sol[1, :]
    m = sol[2, :]
    h = sol[3, :]
    n = sol[4, :]
    I = I₀.(sol.t)

    layout = plot(layout = grid(2, 1))

    plot!(layout[1], sol.t, V, label=nothing, lw=3, ylabel="V", title="Hodgkin Huxley", xticks=nothing)
    plot!(layout[2], sol.t, I, label=nothing, lw=3, ylabel="I₀", color="black")

    @info "displaying"
    display(layout)
end


function run_simulation()
    # initialize params & run simulation
    @info "starting simulation"
    sol = HH(duration=450.0)

    # display plot
    plot_results(sol)

    print("done")
end