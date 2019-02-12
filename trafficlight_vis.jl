
mutable struct TrafficLightVis
    w::TrafficLight
    s::Union{Vec2, Nothing}
    f::Union{Function, Nothing}
    title::Union{String, Nothing}
end

function TrafficLightVis(w::TrafficLight;
                   s=nothing,
                   f=nothing,
                   title=nothing)
    return TrafficLightVis(w, s, f, title)
end

@recipe function f(v::TrafficLightVis)
    xlim --> [v.w.params.initial_state[1], v.w.goal_position]
    ylim --> v.w.params.v_limits
    aspect_ratio --> 1
    title --> something(v.title, "TrafficLightWorld")
    if v.f !== nothing
        @series begin
            f = v.f
            xlim = [v.w.params.initial_state[1], v.w.goal_position]
            ylim = v.w.params.v_limits
            width = v.w.goal_position-v.w.params.initial_state[1]
            height = v.w.params.v_limits[2]-v.w.params.v_limits[1]
            nx = 100
            ny = 100 
            xs = range(xlim[1], stop=xlim[2], length=nx)
            ys = range(ylim[1], stop=ylim[2], length=ny)
            zg = Array{Float64}(undef, ny, nx)
            for i in 1:ny
                for j in 1:nx
                    zg[j,i] = f(Vec2(xs[i], ys[j]))
                end
            end
            color --> cgrad([:red, :white, :green])
            seriestype := :heatmap
            xs, ys, zg
        end
    end
end

Base.show(io::IO, m::MIME, v::TrafficLightVis) = show(io, m, plot(v)) 
Base.show(io::IO, m::MIME"text/plain", v::TrafficLightVis) = println(io, v)