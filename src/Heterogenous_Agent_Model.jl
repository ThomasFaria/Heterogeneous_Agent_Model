module Heterogenous_Agent_Model

using LinearAlgebra, Statistics, Interpolations, Optim, ProgressBars, Printf, QuantEcon

function state_transition(z_next::Float64, skill_next::Float64, age_next::Int64, z::Float64, skill::Float64, age::Float64, Model)
    (; z_chain, skill_chain, age_chain) = Model

    # Get the index of the current/next idiosync shock in the transition matrix
    z_i = z_chain.state_values .== z
    next_z_i = z_chain.state_values .== z_next
    
    # Get the index of the current/next wage/skill in the transition matrix
    skill_i = skill_chain.state_values .== skill
    next_skill_i = skill_chain.state_values .== skill_next

    # Get the index of the current/next age in the transition matrix
    age_i = age_chain.state_values .== age
    next_age_i = age_chain.state_values .== age_next

    π = z_chain.p[z_i, next_z_i] * skill_chain.p[skill_i, next_skill_i] * age_chain.p[age_i, next_age_i]

    return π[:][begin]
end
export state_transition

function a_transition(a::Float64, c::Float64, z::Float64, w::Float64, age::Float64,  Model)
    (; r, Ω) = Model
    a_next = (1 + r) * a + Ω[Int(age)] * w * z - c 
    return a_next
end
export a_transition

function c_transition(a::Float64, a_next::Float64, z::Float64, w::Float64, age::Float64,  Model)
    (; r, Ω) = Model
    c = (1 + r) * a + Ω[Int(age)] * w * z - a_next 
    return c
end
export c_transition

function ubound(a::Float64, z::Float64, w::Float64, age::Float64,  Model)
    (; r, Ω, a_min) = Model
    # Case when you save everything, no consumption only savings
    c = a_min
    ub = (1 + r) * a + Ω[Int(age)] * w * z - c

    if ub < a_min
        ub = a_min
    end
    return [ub]
end
export ubound

function eval_value_function(V::Vector{Float64}, C::Vector{Float64}, Model)
    (; β, z_chain, skill_chain, age_chain, a_vals, n, s_vals, u) = Model
    Tv = zeros(n)
    for s_i in 1:n
        # Get the value of the asset for a given state
        a = s_vals[s_i, 1]
        # Get the value of the idiosync shock for a given state
        z = s_vals[s_i, 2]
        # Get the value of wage for a given state
        w = s_vals[s_i, 3]
        # Get the current age
        age = s_vals[s_i, 4]
        # Get the value of consumption for a given state
        c = C[s_i]
        # Deduce the asset for the following state with the transition rule
        a_next = a_transition(a, c, z, w, age, Model)
        # Initialise the expectation
        Ev_new = 0
        for z_next ∈ z_chain.state_values
            for age_next ∈ age_chain.state_values
                for skill_next ∈ skill_chain.state_values
                    π = state_transition(z_next, skill_next, age_next, z, w, age, Model)
                    if π == 0
                        # No need to interpolate in this case, so we save some time
                    else
                        # # Extract the value function a skill given for the next shock
                        v = V[(s_vals[:,2] .== z_next) .&& (s_vals[:,3] .== skill_next) .&& (s_vals[:,4] .== age_next)]
                        # Interpolate the value function on the asset for the following state
                        v_new = CubicSplineInterpolation(a_vals, v, extrapolation_bc = Line())(a_next[begin])
                        # Compute the expectation
                        Ev_new += v_new * π
                    end
                end
            end
        end
        Tv[s_i] = u(c) + β * Ev_new
    end
    return Tv
end
export eval_value_function

function obj(V::Vector{Float64}, a_new, a::Float64, z::Float64, w::Float64, age::Float64, Model)
    (;z_chain, skill_chain, age_chain, s_vals, a_vals, β, u) = Model
    # Initialise the expectation
    Ev_new = 0
    for z_next ∈ z_chain.state_values
        for age_next ∈ age_chain.state_values
            for skill_next ∈ skill_chain.state_values
                π = state_transition(z_next, skill_next, age_next, z, w, age, Model)
                if π == 0
                    # No need to interpolate in this case, so we save some time
                else
                    # # Extract the value function a skill given for the next shock
                    v = V[(s_vals[:,2] .== z_next) .&& (s_vals[:,3] .== skill_next) .&& (s_vals[:,4] .== age_next)]
                    # Interpolate the value function on the asset for the following state
                    v_new = CubicSplineInterpolation(a_vals, v, extrapolation_bc = Line())(a_new[begin])
                    # Compute the expectation
                    Ev_new += v_new * π
                end
            end
        end
    end
    
    c = c_transition(a, a_new[begin], z, w, age,  Model)
    VF = u(c) + β* Ev_new
    return VF
end
export obj

function bellman_update(V::Vector{Float64}, Model)
    (; n, s_vals, a_min) = Model
    Tv = zeros(n)
    C = zeros(n)
    # Loop through the different states
    for s_i in 1:n
        # Get the value of the asset for a given state
        a = s_vals[s_i, 1]
        # Get the value of the idiosync shock for a given state
        z = s_vals[s_i, 2]
        # Get the value of wage for a given state
        w = s_vals[s_i, 3]
        # Get the age for a given state
        age = s_vals[s_i, 4]

        # Specify lower bound for optimisation
        lb = [a_min]        
        # Specify upper bound for optimisation
        ub = ubound(a, z, w, age, Model)
        # Specify the initial value in the middle of the range
        init = (lb + ub)/2

        # Optimization
        if lb != ub
            Sol = optimize(x -> -obj(V, x, a, z, w, age, Model), lb, ub, init)
            a_new = Optim.minimizer(Sol)[begin]
        else
            a_new = a_min
        end
        # Deduce optimal value function and consumption
        c = c_transition(a, a_new, z, w, age,  Model)

        Tv[s_i] = obj(V, a_new, a, z, w, age, Model) 
        C[s_i] = c
    end
    return (Tv=Tv, C=C)

end
export bellman_update

function solve_PFI(Model; maxit=300, η_tol=1e-5, verbose=false)
    (;n, a_min) = Model

    # First evaluation of the value function 
    V = eval_value_function(zeros(n), zeros(n) .+ a_min, Model)

    η0 = 1.0
    iter = ProgressBar(1:maxit)
    for n in iter
                
        Bellman = bellman_update(V, Model)
        Tv = eval_value_function(Bellman.Tv, Bellman.C, Model)
        
        η = maximum(abs, Tv - V)
    

        λ = η/η0
        η0 = η
        if verbose
            println(n, " : ", η, " : ", λ)
        end
        
        V = Tv

        if η<η_tol
            println("\n Algorithm stopped after iteration ", n, ", with μ = ", λ, "\n")
            return (V=V, C=Bellman.C)
        end
        set_postfix(iter, η=@sprintf("%.8f", η), λ=@sprintf("%.8f", λ))
    end
end
export solve_PFI

function simulate_model(dr::Vector{Float64}, Model; N=1, a0=0.5, T=1000)
    (; z_chain, skill_chain, age_chain, s_vals, a_size) = Model

    sim = zeros(N, T, 5)
    a_init = a0
    for n ∈ ProgressBar(1:N)
        Z = simulate(z_chain, T+1)
        W = simulate(skill_chain, T+1)
        AGE = Float64.(simulate(age_chain, T+1, init = 1))
    
        z0 = Z[begin]
        w0 = W[begin]
        age0 = AGE[begin]
        a0 = a_init
        for t ∈ 1:T
            C = dr[(s_vals[:,2] .== z0) .&& 
                    (s_vals[:,3] .== w0) .&& 
                    (s_vals[:,4] .== age0)
                    ]

            c0 = C[argmin(abs.(s_vals[1:a_size,1] .- a0))]
            sim[n, t, : ] = [a0, z0, w0, age0, c0]

            a0 = a_transition(a0, c0, z0, w0, age0, Model)
            z0 = Z[t+1]
            w0 = W[t+1]
            age0 = AGE[t+1]
        end
    end
    return sim 
end
export simulate_model

function get_asset_from_dr(dr::Vector{Float64}, Model)
    (;s_vals, r, Ω) = Model

    OMEGA = get.(Ref(Ω), Int.(s_vals[:,4]), missing)
    W = s_vals[:, 3]
    Z = s_vals[:, 2]
    A = s_vals[:, 1]
    C = dr
    return OMEGA .* W .* Z + (1 + r) * A  - C

end
export get_asset_from_dr

end
