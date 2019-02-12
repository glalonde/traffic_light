using Random
using LinearAlgebra
using POMDPs
using StaticArrays
using Parameters
using GridInterpolations
using POMDPModelTools
using POMDPModels

export
    TrafficParams,
    TrafficLight

const Vec3 = SVector{3, Float64}
const Vec2 = SVector{2, Float64}

# State is [position, velocity, time_until_green], action is [Accelerate, Cruise, Brake]
@with_kw struct TrafficParams
    dt::Float64 = .1
    period::Float64 = 100
    v_limits::Vec2 = [0.0, 10.0]
    a_limits::Vec2 = [-10.0, 10.0]
    initial_state::Vec2 = [-10, 10]
    discount::Float64 = .95
end

# State is [position, velocity, time_until_green], action is [Accelerate, Cruise, Brake]
 struct TrafficLight <: MDP{Vec3, Symbol}
    params::TrafficParams
    goal_position::Float64
    actions::SVector{3, Symbol}
    function TrafficLight(params::TrafficParams)
        # Goal position is a function of the max velocity and acceleration
        v_max = params.v_limits[2]
        a_max = params.a_limits[2]
        goal_position = v_max * v_max / (2.0 * a_max)
        new(params, goal_position, [:accelerate, :cruise, :brake])
    end
end

function RunningRed(s::Vec3)
    # Running red if position is positive and time until green is positive.
    return s[1] > 0 && s[3] > 0
end

function POMDPs.generate_s(w::TrafficLight, s::Vec3, a::Symbol, rng::AbstractRNG)
    acceleration = 0.0
    if a == :accelerate
        acceleration = w.params.a_limits[2]
    elseif a == :brake
        acceleration = w.params.a_limits[1]
    end
    velocity = s[2] + acceleration * w.params.dt
    velocity = clamp(velocity, w.params.v_limits...)
    return Vec3(s[1] + velocity * w.params.dt, velocity, s[3] - w.params.dt)
end

function POMDPs.reward(w::TrafficLight, s::Vec3, a::Symbol, sp::Vec3)
    if RunningRed(sp)
        return -1000
    end
    return -1
end

function POMDPs.isterminal(w::TrafficLight, s::Vec3)
    # End simulation if we blow through a red light or reach the goal
    return RunningRed(s) || s[1] >= w.goal_position
end

function POMDPs.initialstate(w::TrafficLight, rng::AbstractRNG)
    return Vec3(w.params.initial_state..., w.params.period * rand(rng))
end

POMDPs.discount(w::TrafficLight) = w.params.discount
POMDPs.actions(w::TrafficLight) = w.actions
POMDPs.n_actions(w::TrafficLight) = length(w.actions)

include("trafficlight_solver.jl")
include("trafficlight_vis.jl")