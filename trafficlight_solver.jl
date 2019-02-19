struct GIValue{G <: AbstractGrid}
    grid::G
    gdata::Vector{Float64}
end

evaluate(v::GIValue, s::AbstractVector{Float64}) = interpolate(v.grid, v.gdata, convert(Vector{Float64}, s))

@with_kw mutable struct TrafficWorldSolver{RNG<:AbstractRNG} <: Solver
    max_iters::Int              = 50
    tol::Float64                = 0.01
    value_hist::AbstractVector  = []
    rng::RNG                    = Random.GLOBAL_RNG
end

struct TrafficLightPolicy{V} <: Policy
    actions::Vector{Symbol}
    Qs::Vector{V}
end

# Return an array of values logspaced from min to max, with the denser values closer to pivot
function LogSpaceAround(min::T, pivot::T, max::T, num_vals::Int, base::T = T(10)) where {T<:Real}
    @assert(pivot >= min)
    @assert(pivot <= max)
    @assert(num_vals > 0)
    low_range = log(base, pivot - min + 1)
    hi_range = log(base, max - pivot + 1)
    num_low = round(Int64, low_range / (low_range + hi_range) * num_vals)
    num_hi = num_vals - num_low
    out = Array{T,1}(undef, num_vals)
    low_vals = view(out, 1:num_low - 1)
    hi_vals = view(out, (num_low + 1):num_vals)
    low_vals .= range(low_range, stop=0, length=num_low)[1:end-1]
    out[num_low] = pivot
    hi_vals .= range(0, stop=hi_range, length=num_hi + 1)[2:end]
    @. low_vals = (base ^ low_vals - 1)*-1 + pivot
    @. hi_vals = (base ^ hi_vals - 1) + pivot
    return out
end

function POMDPs.solve(sol::TrafficWorldSolver, w::TrafficLight)
    # Define a discretization for each of the state variables
    @show position_points = LogSpaceAround(w.params.initial_state[1], 0.0, w.goal_position, 45)
    velocity_points = range(w.params.v_limits[1], stop=w.params.v_limits[2], length=30)
    time_points = LogSpaceAround(-w.params.period, 0.0, w.params.period, 60)
    grid = RectangleGrid(position_points, velocity_points, time_points)
    sol.value_hist = []
    data = zeros(length(grid))
    val = GIValue(grid, data)

    for k in 1:sol.max_iters
        newdata = similar(data)
        for i in 1:length(grid)
            s = Vec3(ind2x(grid, i))
            if isterminal(w, s)
                newdata[i] = 0.0
            else
                best_Q = -Inf
                for a in actions(w, s)
                    sp, r = generate_sr(w, s, a, sol.rng)
                    Q = r + discount(w)*evaluate(val, sp)
                    best_Q = max(best_Q, Q)
                end
                newdata[i] = best_Q
            end
        end
        push!(sol.value_hist, val)
        print("\rfinished iteration $k")
        val = GIValue(grid, newdata)
    end

    print("\nextracting policy...     ")

    Qs = Vector{GIValue}(undef,n_actions(w))
    acts = collect(actions(w))
    for j in 1:n_actions(w)
        a = acts[j]
        qdata = similar(val.gdata)
        for i in 1:length(grid)
            s = Vec3(ind2x(grid, i))
            if isterminal(w, s)
                qdata[i] = 0.0
            else
                sp, r = generate_sr(w, s, a, sol.rng)
                Q = r + discount(w)*evaluate(val, sp)
                qdata[i] = Q
            end
        end
        Qs[j] = GIValue(grid, qdata)
    end
    println("done.")

    return TrafficLightPolicy(acts, Qs)
end

function POMDPs.action(p::TrafficLightPolicy, s::AbstractVector{Float64})
    best = action_ind(p, s)
    return p.actions[best]
end

action_ind(p::TrafficLightPolicy, s::AbstractVector{Float64}) = argmax([evaluate(Q, s) for Q in p.Qs])

POMDPs.value(p::TrafficLightPolicy, s::AbstractVector{Float64}) = maximum([evaluate(Q, s) for Q in p.Qs])