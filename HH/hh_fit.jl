"""
    Using the ODEs as a NN layer as outlined here:
        https://towardsdatascience.com/differential-equations-as-a-neural-network-layer-ac3092632255

    the resulting neural-ODE is fitted by a simulated set of dynamics from the HH model
    to infer model parameters
"""

using Revise

include("hh.jl")

import StaticArrays: @MVector
import DiffEqFlux: sciml_train as train
import Flux: ADAM
import Optim: BFGS
import DifferentialEquations: concrete_solve, remake
import ProgressMeter: Progress, next!
import Plots: plot, plot!, display, gr

gr()

# --------------------------------- variables -------------------------------- #
Tstart = 0.0
Tend = 10.0
dT = 0.01

n_iters = 3000


# ----------------------------------- model ---------------------------------- #
# define initial condition and model parameters
u₀ = @MVector [
    -65.0,      # [V] initial voltage
    0.05,       # m
    0.6,        # h
    0.32,       # n
]

p = @MVector [
    1.0,        # SAμF/cm²] - Cm
    120,        # [mS/cm²] - gNa
    36,         # [mS/cm²] - gK
    0.3,        # [mS/cm²] - gL
    50,         # [mV] - ENa
    -77,        # [mV] - EK
    -54.387     # [mV] - EL
]

# setup ODE probem
prob = ODEProblem(hh_dynamics, u₀, (Tstart, Tend), p, dt=dT)
y = Array(solve(prob, Tsit5(); saveat=Tstart:dT:Tend))


"""
    One layer neural network.
    The layer implements an ODE solver simulating the Hudgkin Huxley 
    model given a set of parameters θ.

    The problem has 11 parameters:
        4 initial conditions
        7 model paramters
"""
infer(θ) = Array(
        concrete_solve(
            prob, Tsit5(), u₀, θ, saveat=Tstart:dT:Tend
        )
    )


"""
    Loss function.
    Returns the loss given a set of parameters and other arguments used by
    the callback function during training
"""
function loss(θ)
    ŷ = infer(θ)
    loss = sum(abs2, y[1, :] .- ŷ[1, :])
    loss
end


# Callback function to display progress during training
progress_bar = Progress(n_iters; showspeed=true)

cb = function (p, l)
    next!(progress_bar; showvalues = [("loss", l)])
    return false
end

# train 
# p̂ = @MVector [1.0, 110, 30, 0.4, 55, -88, -40.0]
p̂ = @MVector [
    1.0,        # SAμF/cm²] - Cm
    120,        # [mS/cm²] - gNa
    36,         # [mS/cm²] - gK
    0.3,        # [mS/cm²] - gL
    50,         # [mV] - ENa
    -77,        # [mV] - EK
    -54.387     # [mV] - EL
]
@info "Ready to train"
res = train(loss, p̂, ADAM(), maxiters=n_iters, cb=cb)
# res = train(loss, p̂, BFGS(initial_stepnorm = 0.0001), cb=cb)

@info "Min loss: $(res.minimum) with params: $(round.(p̂')) -> $(round.(res.minimizer')) compared to initial parameters $(round.(p'))"


# plot results
@info "Showing plots"

# solve again to plot results
y = solve(prob,Tsit5(); saveat=Tstart:dT:Tend, dt=dT)
ŷ = solve(remake(prob,p=res.minimizer),Tsit5(),saveat=Tstart:dT:Tend, dt=dT)
y₀ = solve(remake(prob,p=p̂),Tsit5(),saveat=Tstart:dT:Tend, dt=dT)

# plot
plt = plot(y.t, y[1, :], lw=2, color="red", label="y")
plot!(plt, y.t, ŷ[1, :], lw=1, color="black", label="ŷ")
plot!(plt, y.t, y₀[1, :], lw=1, color="blue", label="y₀")
display(plt)

# TODO figure out where instability int raining  loop is coming from