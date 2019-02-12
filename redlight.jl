module TrafficLightEnv

using Reinforce: AbstractEnvironment
using LearnBase: DiscreteSet
using RecipesBase
using Distributions
using Random: seed!


import Reinforce: reset!, actions, finished, step!

export
  TrafficLight,
  reset!,
  step!,
  actions,
  finished,
  f

mutable struct VehicleState
    position::Float64
    velocity::Float64
end

struct Parameters 
    dt::Float64
    period::Float64
    v_min::Float64
    v_max::Float64
    a_min::Float64
    a_max::Float64
    goal_position::Float64
    intial_state::VehicleState
    Parameters() = Parameters(.01, 50, 0, 10, -10, 10, VehicleState(-100, 10))
    function Parameters(dt, period, v_min, v_max, a_min, a_max, initial_state)
        # Goal position is a function of the max velocity and acceleration
        goal_position = v_max * v_max / (2.0 * a_max)
        new(dt, period, v_min, v_max, a_min, a_max, goal_position, initial_state)
    end
end

mutable struct TrafficLight <: AbstractEnvironment
    params::Parameters
    time::Float64
    phase::Float64
    state::VehicleState
    reward::Float64
    seed::Int
end
TrafficLight(seed = -1)  = TrafficLight(Parameters(), 0, 0, VehicleState(0, 0), 0, seed)

function reset!(env::TrafficLight)
    if env.seed >= 0
        seed!(env.seed)
        env.seed = -1
    end

    env.state = env.params.intial_state
    env.time = 0
    # Randomize the phase
    env.phase = rand(Distributions.Uniform(0.0, env.params.period / 2.0))
    env
end

actions(env::TrafficLight, s) = DiscreteSet(1:3)
finished(env::TrafficLight, s′) = env.state.position >= env.params.goal_position

function step!(env::TrafficLight, s::VehicleState, a::Int)
    position = env.state.position
    velocity = env.state.velocity
    acceleration = 0
    if a == 1
        acceleration = env.params.a_min
    elseif a == 3
        acceleration = env.params.a_max
    end

    velocity += acceleration * env.params.dt
    velocity = clamp(velocity, env.params.v_min, env.params.v_max)
    position += velocity
    env.state = VehicleState(position, velocity)
    env.time += env.params.dt
    is_green = sin((2 * pi / env.params.period) * (env.time + env.phase)) > 0
    if is_green && position > 0
        env.reward = -1000
    else 
        env.reward = -env.params.dt
    end
    return env.reward, env.state
end
end