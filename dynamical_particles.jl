using Oceananigans.Models.LagrangianParticleTracking: AbstractParticle
import Oceananigans.Models.LagrangianParticleTracking: particle_u_velocity, particle_v_velocity, particle_w_velocity

struct DynamicalParticle{T} <: AbstractParticle
    x :: T
    y :: T
    z :: T
    u :: T
    v :: T
    w :: T    
    δ :: T
    τ :: T
end

@inline particle_u_velocity(particles::StructArray{DynamicalParticle}, p, up) = particles.u[p]
@inline particle_v_velocity(particles::StructArray{DynamicalParticle}, p, vp) = particles.v[p]
@inline particle_w_velocity(particles::StructArray{DynamicalParticle}, p, wp) = particles.w[p]
