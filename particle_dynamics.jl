using Oceananigans
using Oceananigans.Units
Lx =2π
Ly =2π
#Lx = 1000kilometers # east-west extent [m]
#Ly = 1000kilometers # north-south extent [m]
#Lz = 1kilometers    # depth [m]

grid = RectilinearGrid(size = (200, 200),
                       x = (-Lx, Lx),
                       y = (-Ly, Ly),
                       topology = (Periodic, Periodic, Flat))

ψ=Field{Face,Face,Center}(grid)
ψi(x,y)=2 * (cos(x)+cos(y))
set!(ψ,ψi)
Oceananigans.BoundaryConditions.fill_halo_regions!(ψ)
u=-∂y(ψ)
v= ∂x(ψ)

u = compute!(Field(u))
v = compute!(Field(v))
Oceananigans.BoundaryConditions.fill_halo_regions!((u,v))

using GLMakie
heatmap(ψ)
NP=10000
x=[(rand(Float64)-1/2)*4π/3 for p in 1:NP]
y=[(rand(Float64)-1/2)*4π/3 for p in 1:NP]
z=[0 for p in 1:NP]

model=HydrostaticFreeSurfaceModel(;grid,
                                  velocities=PrescribedVelocityFields(;u,v),
                                  particles=LagrangianParticles(; x, y, z))

contourf(ψ)
scatter!(model.particles.properties.x,model.particles.properties.y)

particle_sim=Simulation(model,Δt=0.001,stop_time=100)
run!(particle_sim)
#                                 particles=LagrangianParticles(; x, y, restitution=1.0, dynamics=no_dynamics, parameters=nothing))



