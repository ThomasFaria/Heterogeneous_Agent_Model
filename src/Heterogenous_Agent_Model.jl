module Heterogenous_Agent_Model

using LinearAlgebra, Statistics, Interpolations, Optim, ProgressBars, Printf

function state_transition(z_next, skill_next, age_next, z, skill, age, Model)
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

    return π[:][1]
end
export state_transition

function a_transition(a, c, w, z, Model)
    (; r) = Model
    a_next = (1 + r) * a + w * z - c 
    return a_next
end
export a_transition

function eval_value_function(V, C, Model)
    (; r, β, z_chain, skill_chain, age_chain, a_vals, n, s_vals, u, U) = Model
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
        a_next = a_transition(a, c, w, z, Model)
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
                        v_new = CubicSplineInterpolation(a_vals, v, extrapolation_bc = Line())(a_next[1])
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

function obj(V, a_new, a, z, w, age, Model)
    (;z_chain, skill_chain, age_chain, s_vals, a_vals, r, β, u, U) = Model
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
                    v_new = CubicSplineInterpolation(a_vals, v, extrapolation_bc = Line())(a_new[1])
                    # Compute the expectation
                    Ev_new += v_new * π
                end
            end
        end
    end

    VF = u(a * (1+r) + z * w - a_new[1]) + β* Ev_new
    return VF
end
export obj

function bellman_update(V, Model)
    (; r, a_min, n, s_vals, ϵ) = Model
    
    Tv = zeros(n)
    C = ones(n)*a_min
    # Loop through the different states
    Threads.@threads for s_i in 1:n
        # Get the value of the asset for a given state
        a = s_vals[s_i, 1]
        # Get the value of the idiosync shock for a given state
        z = s_vals[s_i, 2]
        # Get the value of wage for a given state
        w = s_vals[s_i, 3]
        # Get the age for a given state
        age = s_vals[s_i, 4]
        # Get the value of consumption for a given state
        c = C[s_i]

        # Specify lower bound for optimisation
        lb = zeros(1) .+ ϵ        
        # Specify upper bound for optimisation
        ub = [a * (1+r) + z * w] .- ϵ
        # Specify the initial value in the middle of the range
        init = (lb + ub)/2

        # Optimization
        Sol = optimize(x -> -obj(V, x, a, z, w, age, Model), lb, ub, init)
        a_new = Optim.minimizer(Sol)[1]

        # Deduce optimal value function and consumption
        Tv[s_i] = obj(V, a_new, a, z, w, age, Model) 
        C[s_i] =  a * (1+r) + z * w - a_new

    end
    return (Tv=Tv, C=C)

end
export bellman_update

function solve_PFI(Model; maxit=300, η_tol=1e-8, verbose=false)
    (;n) = Model

    # Initialize the value function
    V = zeros(n)

    # Initialize the consumption function
    C = zeros(n)
    
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
        set_postfix(iter, η=@sprintf("%.8f", η))
    end
end
export solve_PFI

end
