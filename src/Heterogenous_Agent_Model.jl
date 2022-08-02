module Heterogenous_Agent_Model

using LinearAlgebra, Statistics, Interpolations, Optim, ProgressBars, Printf, QuantEcon, CSV, NLsolve, AxisArrays


function import_aging_prob(age_min::Int64, age_max::Int64)
    data = CSV.read("data/LifeTables.csv", NamedTuple)
    ψ = data.Prob2survive
    # After a certain age death is certain
    ψ[age_max+1:end] .= 0

    return ψ[age_min+1:end]
end
export import_aging_prob


function pop_distrib(μ_1::Vector{Float64}, ψ::Vector{Float64}, find_root::Bool)
    μ = zeros(size(ψ))
    μ[begin] = μ_1[1]

    for (index, value) in enumerate(ψ[2:end])
        μ[index+1] = value* μ[index]
    end

    if find_root
        return sum(μ) - 1
    else
        return μ
    end
end

function get_pop_distrib(ψ::Vector{Float64})
    # Find first share that assure a sum equal to 1
    sol = nlsolve(x -> pop_distrib(x, ψ, true), [0.02])
    μ = pop_distrib(sol.zero, ψ, false)
    return μ
end
export get_pop_distrib


function get_efficiency(age_min::Int64, age_max::Int64)
    ϵ =  [exp(-1.46 + 0.070731*j - 0.00075*j^2) for j in age_min:age_max]
    return ϵ
end
export get_efficiency


function get_soc_sec_benefit(ϵ::Vector{Float64}, h::Float64, w::Float64, j_star::Int64, J::Int64, Policy::NamedTuple)
    (; θ) = Policy
    W = w * h * ϵ
    b  = zeros(J)
    b[j_star:end] .= θ * mean(W)
    return b
end
export get_soc_sec_benefit

function get_wages(ϵ::Vector{Float64}, h::Float64, w::Float64, Policy::NamedTuple)
    (; ξ) = Policy
    W = zeros(size(ϵ, 1), 2)

    W[:,1] .= w * h * ξ
    W[:,2] = w * h * ϵ
    return W
end
export get_wages

function get_dispo_income(W::Matrix{Float64}, b::Vector{Float64}, j_star::Int64, Policy::NamedTuple)
    (; τ_ssc, τ_u) = Policy
    q = zeros(size(b, 1), 2)

    q[begin:j_star,1] = W[:,1]
    q[begin:j_star,2] = (1 - τ_ssc - τ_u) * W[:,2]
    q[j_star+1:end,:] .= b[j_star+1:end]
    return q
end
export get_dispo_income

function state_transition_OLG(z_next::Symbol, z::Symbol, Params::NamedTuple)
    (; z_chain) = Params

    # Get the index of the current/next idiosync shock in the transition matrix
    z_i = z_chain.state_values .== z
    next_z_i = z_chain.state_values .== z_next

    π = z_chain.p[z_i, next_z_i]
    return π[:][begin]
end
export state_transition_OLG

function get_state_value_index(mc::MarkovChain, value)
    idx = findall(mc.state_values .== value)
    return idx[begin]
end 
export get_state_value_index

function c_transition_OLG(a_past::Float64, a::Vector{Float64}, z::Symbol, age::Int64,  Params::NamedTuple)
    (; r, q, z_chain) = Params
    idx = get_state_value_index(z_chain, z)

    c = (1 + r) * a_past + q[age, idx] - a[begin] 
    return c
end
export c_transition_OLG

function ubound(a_past::Float64, z::Symbol, age::Int64,  Params::NamedTuple)
    (; r, q, z_chain, a_min) = Params
    idx = get_state_value_index(z_chain, z)

    # Case when you save everything, no consumption only savings
    c =  a_min
    ub = (1 + r) * a_past + q[age, idx] - c
    ub = ifelse(ub <= a_min, a_min, ub)

    return [ub]
end
export ubound

function obj_OLG(V::AxisArray{Float64, 3}, a::Vector{Float64}, a_past::Float64, z::Symbol, age::Int64, Params::NamedTuple)
    (; ψ, β, z_chain, a_vals, β, u, J, j_star) = Params
    c = c_transition_OLG(a_past, a, z, age, Params)

    if age == J
        VF = u(c) 
    elseif (age < J) & (age >= j_star - 1)
        # # Extract the value function (z is a placeholder here)
        v = V[Age = age + 1, Z = z]
        # Interpolate the value function on the asset for the following state
        v_new = CubicSplineInterpolation(a_vals, v, extrapolation_bc = Line())(a[begin])
        VF = u(c) + β * ψ[age + 1] * v_new
    else
        # Initialise the expectation
        Ev_new = 0
        for z_next ∈ z_chain.state_values
            π = state_transition_OLG(z_next, z, Params)
            # # Extract the value function a skill given for the next shock
            v = V[Age = age + 1, Z = z_next]
            # Interpolate the value function on the asset for the following state
            v_new = CubicSplineInterpolation(a_vals, v, extrapolation_bc = Line())(a[begin])
            # Compute the expectation
            Ev_new += v_new * π
        end
        VF = u(c) + β* ψ[age + 1] * Ev_new
    end
    return VF
end
export obj_OLG

function get_dr(Params::NamedTuple)
    (; β, z_chain, z_size, a_size, a_vals, a_min, β,  J) = Params

    V = AxisArray(zeros(a_size, z_size, J);
                    a = 1:a_size,
                    Z = (z_chain.state_values),
                    Age = 1:J
            )

    A = AxisArray(zeros(a_size, z_size, J);
                        a = 1:a_size,
                        Z = (z_chain.state_values),
                        Age = 1:J
                )
    C = AxisArray(zeros(a_size, z_size, J);
        a = 1:a_size,
        Z = (z_chain.state_values),
        Age = 1:J
    )

    # Loop over ages recursively
    iter = ProgressBar(J:-1:1)
    for j ∈ iter
        # Loop over past assets
        for a_i ∈ 1:a_size
            a = a_vals[a_i]
            # Loop over idiosync shock
            for z ∈ z_chain.state_values
                # Specify lower bound for optimisation
                lb = [a_min]        
                # Specify upper bound for optimisation
                ub = ubound(a, z, j, Params)
                # Specify the initial value in the middle of the range
                init = (lb + ub)/2
                # Optimization
                Sol = optimize(x -> -obj_OLG(V, x, a, z, j, Params), lb, ub, init)
                a_new = Optim.minimizer(Sol)

                # Deduce optimal value function, consumption and asset
                V[Age = j, Z = z, a = a_i] = obj_OLG(V, a_new, a, z, j, Params)
                C[Age = j, Z = z, a = a_i] = c_transition_OLG(a, a_new, z, j,  Params)
                A[Age = j, Z = z, a = a_i] = a_new[begin]

            end
        end
    end
    return (V = V, C = C, A = A)
end
export get_dr

function get_r(K, L, Firm_pm::NamedTuple)
    (;α, A, δ) = Firm_pm
    return A * α * (L/K)^(1-α) - δ
end

function get_w(K, L, Firm_pm::NamedTuple)
    (;α, A) = Firm_pm
    return A * (1-α) * (K/L)^α
end


### Old model

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
