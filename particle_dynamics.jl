using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: architecture
using Oceananigans.Architectures: device
using Oceananigans.Fields: interpolate
using KernelAbstractions: @kernel, @index
using StructArrays
using GLMakie
using JLD2

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

u = XFaceField(grid) # compute!(Field(u))
v = YFaceField(grid) # compute!(Field(v))
w = ZFaceField(grid) 

set!(u, (x, y) -> rand() - 0.5)
set!(v, (x, y) -> rand() - 0.5)

Oceananigans.BoundaryConditions.fill_halo_regions!((u,v))

using GLMakie
heatmap(ψ)

NP=10000
xp=Float64[(rand(Float64)-1/2)*4π/3 for p in 1:NP]
yp=Float64[(rand(Float64)-1/2)*4π/3 for p in 1:NP]
zp=Float64[0 for p in 1:NP]

up=Float64[0 for p in 1:NP]
vp=Float64[0 for p in 1:NP]
wp=Float64[0 for p in 1:NP]

δ=Float64[0.9 for p in 1:NP]
τ=Float64[1.0 for p in 1:NP]

particle_struct = StructArray{DynamicalParticle}((xp, yp, zp, up, vp, wp, δ, τ))

const f = Face()
const c = Center()

@kernel function _nonlinear_dynamics!(particles, grid, velocities, auxiliary_fields, Δt)
    p = @index(Global, Linear)
    u, v, w = velocities

    U_∇u = auxiliary_fields.U_∇u
    U_∇v = auxiliary_fields.U_∇v
    U_∇w = auxiliary_fields.U_∇w

    @inbounds begin
       δ = particles.δ[p]    
       τ = particles.τ[p]
       x = particles.x[p]
       y = particles.y[p]
       z = particles.z[p]

       up = particles.u[p]
       vp = particles.v[p]
       wp = particles.w[p]

       uf = interpolate((x, y, z), u, (f, c, c), grid)
       vf = interpolate((x, y, z), v, (c, f, c), grid)
       wf = interpolate((x, y, z), w, (c, c, f), grid)

       U_∇uf = interpolate((x, y, z), U_∇u, (f, c, c), grid)
       U_∇vf = interpolate((x, y, z), U_∇v, (c, f, c), grid)
       U_∇wf = interpolate((x, y, z), U_∇w, (c, c, f), grid)

       particles.u[p] += Δt * (δ * U_∇uf - (up - uf) / τ)
       particles.v[p] += Δt * (δ * U_∇vf - (vp - vf) / τ)
       particles.w[p] += Δt * (δ * U_∇wf - (wp - wf) / τ)
    end
end

function nonlinear_dynamics(particles, model, Δt)

    grid = model.grid
    arch = architecture(grid)
    dev  = device(arch)
    Np   = length(particles)
    auxiliary_fields = model.auxiliary_fields
    _nonlinear_dynamics!(dev, 16, Np)(particles.properties, model.grid, model.velocities, auxiliary_fields, Δt)

    return nothing
end

using Oceananigans.Advection: div_𝐯u, div_𝐯u, div_𝐯w

advection = WENO()

# U_∇u = KernelFunctionOperation{Face, Center, Center}(div_𝐯u, grid, advection, (; u, v, w), u)
# U_∇v = KernelFunctionOperation{Face, Center, Center}(div_𝐯u, grid, advection, (; u, v, w), v)
# U_∇w = KernelFunctionOperation{Face, Center, Center}(div_𝐯u, grid, advection, (; u, v, w), w)

U_∇u = compute!(Field(KernelFunctionOperation{Face, Center, Center}(div_𝐯u, grid, advection, (; u, v, w), u)))
U_∇v = compute!(Field(KernelFunctionOperation{Face, Center, Center}(div_𝐯u, grid, advection, (; u, v, w), v)))
U_∇w = compute!(Field(KernelFunctionOperation{Face, Center, Center}(div_𝐯u, grid, advection, (; u, v, w), w))) 

model=NonhydrostaticModel(;grid, advection,
                           velocities=(; u, v, w),
                           particles=LagrangianParticles(particle_struct; dynamics=nonlinear_dynamics),
                           auxiliary_fields = (; U_∇u, U_∇v, U_∇w))

particle_sim=Simulation(model, Δt=0.005, stop_time=100)

function compute_derivatives!(sim)
   auxiliary_fields = sim.model.auxiliary_fields

   U_∇u = auxiliary_fields.U_∇u
   U_∇v = auxiliary_fields.U_∇v
   U_∇w = auxiliary_fields.U_∇w
   
   compute!(U_∇u)
   compute!(U_∇v)
   compute!(U_∇w)

   return nothing
end

add_callback!(particle_sim, compute_derivatives!, IterationInterval(1)) 

progress(sim) = @info "Time: ", Oceananigans.Utils.prettytime(sim.model.clock.time) 
add_callback!(particle_sim, progress, IterationInterval(100))

ζ = ∂x(v) - ∂y(u)

particle_sim.output_writers[:particles] = JLD2Writer(model, (; p = model.particles, ζ),
                                                     filename = "particles.jld2",
                                                     schedule = IterationInterval(10),
                                                     overwrite_existing = true)

run!(particle_sim)

contourf(ψ)
scatter!(model.particles.properties.x,model.particles.properties.y)

ζt = FieldTimeSeries("particles.jld2", "ζ")

file    = jldopen("particles.jld2")
iter    = Observable(1)
indices = keys(file["timeseries/t"])

xp = @lift(file["timeseries/p/" * indices[$iter]].x)
yp = @lift(file["timeseries/p/" * indices[$iter]].y)
δp = @lift(file["timeseries/p/" * indices[$iter]].δ)
up = @lift(file["timeseries/p/" * indices[$iter]].u)
vp = @lift(file["timeseries/p/" * indices[$iter]].v)
sp = @lift($up.^2 + $vp.^2)
ζn = @lift(ζt[$iter])

fig = Figure()
ax  = Axis(fig[1, 1])
contourf!(ζn, colormap = :greys)
scatter!(xp, yp, color = δp, colormap = :magma)
xlims!(ax, (-2π, 2π))
ylims!(ax, (-2π, 2π))

record(fig, "particle_video.mp4", 1:10:length(indices)) do i
    @info "recording $i of $(length(indices))"
    iter[] = i
end

close(file)
