using Base: @kwdef
using StaticArrays: @MVector, SA
using Plots
using DifferentialEquations: ODEProblem, solve

gr()

"""
    Julia implementation of the Hodgkin-Huxley model of single neuron membrane dynamics.
    Adapted from a tutorial in Python:
    https://hodgkin-huxley-tutorial.readthedocs.io/en/latest/_static/Hodgkin%20Huxley.html
"""

# ----------------------------------- parms ---------------------------------- #
@kwdef mutable struct SimulationParams
    I₀::Float64 = 0.0  # input current
    dt::Float64 = 0.01  # [ms]
end

@kwdef struct ModelParams
    # capacitance
    Cm = 1.0  # [μF/cm²]

    # max conductances
    gNa = 120  # [mS/cm²]
    gK = 36  # [mS/cm²]
    gL = 0.3  # [mS/cm²]

    # Nerst potentials
    ENa = 50  # [mV]
    EK = -77  # [mV]
    EL = 54.387  # [mV]
end

# -------------------------- channel gating kinetics ------------------------- #
alphaM(V::Float64) = 0.1*(V+40.0)/(1.0 - exp(-(V+40.0) / 10.0))

betaM(V::Float64) = 4.0*exp(-(V+65.0) / 18.0)

alphaH(V::Float64) = 0.07*exp(-(V+65.0) / 20.0)

betaH(V::Float64) = 1.0/(1.0 + exp(-(V+35.0) / 10.0))

alphaN(V::Float64) = 0.01*(V+55.0)/(1.0 - exp(-(V+55.0) / 10.0))

betaN(V::Float64) = 0.125*exp(-(V+65) / 80.0)


# ---------------------------- currents functions ---------------------------- #
I_Na(V::Float64, m::Float64, h::Float64, gNa::Float64, ENa::Float64) = gNa * m^3 * h * (V - ENa)
I_K(V::Float64, n::Float64, gK::Float64, EK::Float64) = gK * n^4 * (V-EK)
I_L(V::Float64, gL::Float64, EL::Float64) = gL * (V - EL)

# --------------------------------- HH model --------------------------------- #
"""
    HH model simulation
"""
function hh_dynamics(u, p, t)
    V₀, m₀, h₀, n₀ = u
    Cm, gNa, gK, gL, ENa, EK, EL, I₀ = p

    V₁ = (I₀ - I_Na(V₀, m₀, h₀, gNa, ENa) - I_K(V₀, n₀, gK, EK) - I_L(V₀, gL, EL))/Cm
    
    m₁ = alphaM(V₀)*(1-m₀) - betaM(V₀)*m₀
    h₁ = alphaH(V₀)*(1-h₀) - betaH(V₀)*h₀
    n₁ = alphaN(V₀)*(1-n₀) - betaN(V₀)*n₀
    SA[V₁, m₁, h₁, n₁]
end

function HH(sp::SimulationParams, mp::ModelParams; duration::Float64)
    # define an initial state and a params array
    state = SA[
        -70.0,  # [V] initial voltage
        0.05,  # m
        0.54,  # h
        0.34,  # n
    ]
    params = SA[
        mp.Cm,
        mp.gNa,
        mp.gK,
        mp.gL,
        mp.ENa,
        mp.EK,
        mp.EL,
        sp.I₀,
    ]
 
    # solve problem
    tspan = (0.0, duration)
    prob = ODEProblem(hh_dynamics, state, tspan, params)
    sol = solve(prob; saveat=sp.dt)

    return sol
end


function plot_results(sol)
    @info "Showing plots"

    V = sol[1, :]
    m = sol[2, :]
    h = sol[3, :]
    n = sol[4, :]
    layout = plot(layout = grid(4, 1))

    plot!(layout[1], sol.t, V, label=nothing, lw=3, ylabel="V")
    plot!(layout[2], sol.t, m, color="red", label=nothing, lw=3, ylabel="m")
    plot!(layout[3], sol.t, h, color="green", label=nothing, lw=3, ylabel="h")
    plot!(layout[4], sol.t, n, color="black", label=nothing, xlabel="Time [s]", lw=3, ylabel="n")

    @info "displaying"
    display(layout)
end

# initialize params & run simulation
@info "starting simulation"
sp = SimulationParams(I₀=10.0)
mp = ModelParams()

sol = HH(sp, mp; duration=450.0)

# display plot
plot_results(sol)

print("done")