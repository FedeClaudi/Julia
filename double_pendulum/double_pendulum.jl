using DifferentialEquations, Plots
using BenchmarkTools, StaticArrays


function dynamics(u,p,t)
    g, l₁, l₂, m₁, m₂ = p
    θ₁, ω₁, θ₂, ω₂ = u

    δ = θ₂ - θ₁
    den1 = (m₁+m₂) * l₁ - m₂ * l₁ * cos(δ) * cos(δ)
    den2 = (l₂/l₁) * den1

    du1 = ω₁
    du2 = ((m₂ * l₁ * ω₁ * ω₁ * sin(δ) * cos(δ)
                + m₂ * g * sin(θ₂) * cos(δ)
                + m₂ * l₂ * ω₂ * ω₂ * sin(δ)
                - (m₁+m₂) * g * sin(θ₁))
            / den1)
    du3 = ω₂

    du4 = ((- m₂ * l₂ * ω₂ * ω₂ * sin(δ) * cos(δ)
                + (m₁+m₂) * g * sin(θ₁) * cos(δ)
                - (m₁+m₂) * l₁ * ω₁ * ω₁ * sin(δ)
                - (m₁+m₂) * g * sin(θ₂))
            / den2)
    SA[du1,du2,du3,du4]
end


function make()
    # pendulum variables
    g = 9.8  # acceleration due to gravity, in m/s^2
    l₁ = 1.0  # length of pendulum 1 in m
    l₂ = 1.0  # length of pendulum 2 in m
    m₁ = 1.0  # mass of pendulum 1 in kg
    m₂ = 1.0  # mass of pendulum 2 in kg
    params = SA[g, l₁, l₂, m₁, m₂]

    # solve double pendulum
    tspan = (0.0, 100.0)
    pendulum = SA[120.0, 0.0, -10.0, 0.0]
    prob = ODEProblem(dynamics, pendulum, tspan, params)
    sol = solve(prob; saveat=.01)

    return params, sol
end

function plot_solution(params, sol)
    g, l₁, l₂, m₁, m₂ = params

    # get position of massess
    x1 = @. l₁*sin(sol[1, :])
    y1 = @. -l₁*cos(sol[1, :])

    x2 = @. l₂*sin(sol[3, :]) + x1
    y2 = @. -l₂*cos(sol[3, :]) + y1

    # plot
    plt = plot(x1, y1)
    plot!(plt, x2, y2)
    display(plt)
end

function animate_solution(params, sol)
    g, l₁, l₂, m₁, m₂ = params

    # get position of massess
    x1 = @. l₁*sin(sol[1, :])
    y1 = @. -l₁*cos(sol[1, :])

    x2 = @. l₂*sin(sol[3, :]) + x1
    y2 = @. -l₂*cos(sol[3, :]) + y1

    animate
    anim = @animate for i = 1:length(x1)
        scatter([x1[i]], [y1[i]], legend=false)
        scatter([x2[i]], [y2[i]], legend=false)
    end
    
    println("Saving to video")
    mov(anim, "tutorial_anim_fps30.gif", fps = 30)
end

# make twice to get run time 
println("\n\nStart")
@time make()
params, solution = @time make()

println("Solution size: ", size(solution))

# # plot_solution(params, solution)
# # println("done plotting, making animation")

# # animate_solution(params, solution)
println("all done")
