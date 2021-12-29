using Revise
using Base: @kwdef
using StaticArrays: @MVector
using Plots

gr()

"""
    Julia implementation of the Hodgkin-Huxley model of single neuron membrane dynamics.
    Adapted from a tutorial in Python:
        https://mark-kramer.github.io/Case-Studies-Python/HH.html
    original Python code at:
        https://github.com/Mark-Kramer/Case-Studies-Python
"""

# ----------------------------------- parms ---------------------------------- #
@kwdef struct SimulationParams
    I₀::Float64 = 0.0  # input current
    duration::Float64 = 1  # [ms]
    dt::Float64 = 0.01  # [ms]
end

@kwdef struct ModelParams
    # conductances
    gNa₀ = 120  # [mS/cm²]
    gK₀ = 36  # [mS/cm²]
    gL₀ = 0.3  # [mS/cm²]

    # potentials
    ENa = 115  # [mV]
    EK = -12  # [mV]
    EL = 10.6  # [mV]
end

# ----------------------------- gating functions ----------------------------- #
alphaM(V::Float64) = (2.5-0.1*(V+65)) / (exp(2.5-0.1*(V+65)) -1)

betaM(V::Float64) = 4*exp(-(V+65)/18)

alphaH(V::Float64) = 0.07*exp(-(V+65)/20)

betaH(V::Float64) = 1/(exp(3.0-0.1*(V+65))+1)

alphaN(V::Float64) = (0.1-0.01*(V+65)) / (exp(1-0.1*(V+65)) -1)

betaN(V::Float64) = 0.125*exp(-(V+65)/80)


# --------------------------------- HH model --------------------------------- #
"""
    HH model simulation
"""
function HH(sp::SimulationParams, mp::ModelParams)
    T = Int64(ceil(sp.duration/sp.dt))  # num sim steps

    # pre allocate values
    t = (1 : T) * sp.dt  # time in [ms]
    V = @MVector zeros(T)
    m = @MVector zeros(T)
    h = @MVector zeros(T)
    n = @MVector zeros(T)

    # set initial values
    V[1]=-70.0
    m[1]=0.05
    h[1]=0.54
    n[1]=0.34

    # simulation loop
    for i in 1:T-1
        Vₜ = view(V, i)[1]
        mₜ = view(m, i)[1]
        hₜ = view(h, i)[1]
        nₜ = view(n, i)[1]


        V[i+1] = Vₜ + sp.dt*(
            mp.gNa₀ * mₜ^3 * hₜ * (mp.ENa-(Vₜ+65)) + 
            mp.gK₀ * nₜ^4 * (mp.EK-(Vₜ+65)) + 
            mp.gL₀ * (mp.EL-(Vₜ+65)) + sp.I₀
        )
        
        m[i+1] = mₜ + sp.dt*(alphaM(Vₜ)*(1-mₜ) - betaM(Vₜ)*mₜ)
        h[i+1] = hₜ + sp.dt*(alphaH(Vₜ)*(1-hₜ) - betaH(Vₜ)*hₜ)
        n[i+1] = nₜ + sp.dt*(alphaN(Vₜ)*(1-nₜ) - betaN(Vₜ)*nₜ)
    end

    return V, m, h, n, t
end


function plot_results(V, m, h, n, t)
    @info "Showing plots"
    layout = plot(layout = grid(4, 1))

    plot!(layout[1], t, V, label=nothing, lw=3, ylabel="V")
    plot!(layout[2], t, m, color="red", label=nothing, lw=3, ylabel="m")
    plot!(layout[3], t, h, color="green", label=nothing, lw=3, ylabel="h")
    plot!(layout[4], t, n, color="black", label=nothing, xlabel="Time [s]", lw=3, ylabel="n")

    display(layout)
end

# TODO use ODEs integration to get results instead of manual loops

# run simulation
@info "starting simulation"
sp = SimulationParams(I₀=0.0, duration=10.0)
mp = ModelParams()
V, m, h, n, t = @time HH(sp, mp)

# display plot
plot_results(V, m, h, n, t)

print("done")