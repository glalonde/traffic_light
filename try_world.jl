using Revise
using POMDPSimulators
using Interact
using Plots
plotly()
include("trafficlight.jl")

pomdp = TrafficLight(TrafficParams())
sol = TrafficWorldSolver(max_iters=50)
@time policy = solve(sol, pomdp);
hr = HistoryRecorder()
history = simulate(hr, pomdp, policy)
println(length(history))
for (s, a, r, sp) in eachstep(history, "(s, a, r, sp)")    
    println("reward $r received when state $sp was reached after action $a was taken in state $s")
end

# Show how the value function evolves turning iteration
@manipulate for i in 1:length(sol.value_hist)
    v = sol.value_hist[i]
    plot(TrafficLightVis(pomdp, f=s->evaluate(v, Vec3(s..., 10.0))))
end

# Show the policy for various times
@manipulate for t in range(-10,stop=10, step=.1)
    plot(TrafficLightVis(pomdp, f=s->action_ind(policy, Vec3(s..., t)), title="Policy"))
end