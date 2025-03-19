using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: architecture
using Oceananigans.Architectures: device
using KernelAbstractions: @kernel, @index
using StructArrays

include("dynamical_particles.jl")

Lx = 2π
Ly = 2π

grid = RectilinearGrid(size = (200, 200),
                       x = (-Lx, Lx),
                       y = (-Ly, Ly),
                       topology = (Periodic, Periodic, Flat))

ψ = Field{Face, Face, Center}(grid)

ψi(x, y) =2 * (cos(x)+cos(y))

set!(ψ, ψi)

Oceananigans.BoundaryConditions.fill_halo_regions!(ψ)
u = - ∂y(ψ)
v =   ∂x(ψ)

u = compute!(Field(u))
v = compute!(Field(v))

Oceananigans.BoundaryConditions.fill_halo_regions!((u,v))

using GLMakie
heatmap(ψ)

NP=10000
x=Float64[(rand(Float64)-1/2)*4π/3 for p in 1:NP]
y=Float64[(rand(Float64)-1/2)*4π/3 for p in 1:NP]
z=Float64[0 for p in 1:NP]

u=Float64[0 for p in 1:NP]
v=Float64[0 for p in 1:NP]
w=Float64[0 for p in 1:NP]

δ=Float64[0.9 for p in 1:NP]
τ=Float64[1.0 for p in 1:NP]

particle_struct = StructArray{DynamicalParticle}((x, y, z, u, v, w, δ, τ))

@kernel function _nonlinear_dynamics!(particles, Δt)
    p = @index(Global, Linear)
    @inbounds begin
       δ = particles.δ[p]    
       τ = particles.τ[p]
       x = particles.x[p]
       y = particles.y[p]

       up = particles.u[p]
       vp = particles.v[p]
       uf =   2 * sin(y)
       vf = - 2 * sin(x)
       U_∇u = - 4 * sin(x) * cos(y)
       U_∇v = - 4 * cos(x) * sin(y)

       particles.u[p] += Δt * (δ * U_∇u - (up - uf) / τ)
       particles.v[p] += Δt * (δ * U_∇v - (vp - vf) / τ)
    end
end

function nonlinear_dynamics(particles, model, Δt)

    grid = model.grid
    arch = architecture(grid)
    dev  = device(arch)
    Np   = length(particles)
    _nonlinear_dynamics!(dev, 16, Np)(particles.properties, Δt)

    return nothing
end

model=HydrostaticFreeSurfaceModel(;grid,
                                  velocities=PrescribedVelocityFields(; u, v),
                                  particles=LagrangianParticles(particle_struct; dynamics=nonlinear_dynamics))

particle_sim=Simulation(model, Δt=0.005, stop_time=100)

progress(sim) = @info "Time: ", Oceananigans.Utils.prettytime(sim.model.clock.time) 
add_callback!(particle_sim, progress, IterationInterval(100))

particle_sim.output_writers[:particles] = JLD2Writer(model, (; p = model.particles),
                                                     filename = "particles.jld2",
                                                     schedule = IterationInterval(10),
                                                     overwrite_existing = true)

run!(particle_sim)

contourf(ψ)
scatter!(model.particles.properties.x,model.particles.properties.y)

file    = jldopen("particles.jld2")
iter    = Observable(1)
indices = keys(file["timeseries/t"])

xp = @lift(file["timeseries/p/" * indices[$iter]].x)
yp = @lift(file["timeseries/p/" * indices[$iter]].y)
δp = @lift(file["timeseries/p/" * indices[$iter]].δ)
up = @lift(file["timeseries/p/" * indices[$iter]].u)
vp = @lift(file["timeseries/p/" * indices[$iter]].v)
sp = @lift($up.^2 + $vp.^2)

fig = Figure()
ax  = Axis(fig[1, 1])
contourf!(ψ, colormap = :greys)
scatter!(xp, yp, color = δp, colormap = :magma)
xlims!(ax, (-2π, 2π))
ylims!(ax, (-2π, 2π))

record(fig, "particle_video.mp4", 1:10:length(indices)) do i
    @info "recording $i of $(length(indices))"
    iter[] = i
end

close(file)