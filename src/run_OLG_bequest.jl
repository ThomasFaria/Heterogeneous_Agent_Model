using Heterogenous_Agent_Model, QuantEcon, LaTeXStrings, Parameters, Plots, Serialization, StatsPlots, AxisArrays

Policy = @with_kw (
                    ξ = 0.4,
                    θ = 0.3,
                    τ_ssc = θ * (sum(HHs.μ[HHs.j_star:end]) * mean(HHs.ϵ)) / (sum(HHs.μ[begin:HHs.j_star-1] .* HHs.ϵ)  * 0.94),
                    τ_u = (0.06/0.94) * ξ
)    

Households = @with_kw ( 
                    death_age = 85,
                    retirement_age = 65,
                    age_start_work = 21,
                    h = 0.45,
                    ρ = 0.012,
                    j_star = retirement_age - age_start_work + 1,
                    J = death_age - age_start_work + 1,
                    ψ = import_aging_prob(age_start_work, death_age), # Probabilities to survive
                    μ = get_pop_distrib(ψ, ρ), # Population distribution
                    ϵ = get_efficiency(age_start_work, retirement_age - 1), #Efficiency index
                    γ = 2., # Constant relative risk aversion (consumption utility)
                    Σ = 1., # Constant relative risk aversion (asset utility)
                    β = 1.011,
                    z_chain = MarkovChain([0.06 0.94;
                                           0.06 0.94], 
                                        [:U; :E]),
                    a_min = 1e-10,
                    a_max = 15.,
                    a_size = 100,
                    a_vals = range(a_min, a_max, length = a_size),
                    z_size = length(z_chain.state_values),
                    u = γ == 1 ? c -> log(c) : c -> (c^(1 - γ)) / (1 - γ),
                    U = Σ == 1 ? a -> log(a) : a -> (a^(1 - Σ)) / (1 - Σ),
)

Firms = @with_kw ( 
    α = 0.36,
    Ω = 1.2082, #1.3193,
    δ = 0.08,
)

# Initial values
Firm = Firms();
HHs = Households();
Policies = Policy()

using LinearAlgebra, Statistics, Interpolations, Optim, ProgressBars, Printf, QuantEcon, CSV, NLsolve, AxisArrays, Distributions, DataStructures, Parameters

λ = deserialize("data/distrib_03.dat")
function bequest_distrib(λ::NamedTuple, Params::NamedTuple)
    (; a_size, j_star, J, ψ) = Params
    υ = AxisArray(zeros(a_size, 2, 2, J-20);
        b = 1:a_size,
        ϕ_next = [:I, :NI],
        ϕ = [:I, :NI],
        Age = 1:J-20
    )

    for j ∈ 1:J-20
        age_parents = j + 20
        # Already inherited so for sure no more bequest and state doesn't change 
        υ[ϕ = :I, Age = j, ϕ_next = :I, b = 1] = 1.
        # Not inherited yet 
            ## Parents are still alive
            υ[ϕ = :NI, Age = j, ϕ_next = :NI, b = 1] = 1. #ψ[age_parents]
            ## Parents just dead
                if age_parents < j_star
                    ### Parents are not retired
                    υ[ϕ = :NI, Age = j, ϕ_next = :I] = sum(λ.λ_a[Age = age_parents], dims = 2) #(1 - ψ[age_parents]) .* sum(λ.λ_a[Age = age_parents], dims = 2)
                else
                    ### Parents are retired
                    υ[ϕ = :NI, Age = j, ϕ_next = :I] = λ.λ_r[Age = age_parents - (j_star - 1)] #(1 - ψ[age_parents]) .* λ.λ_r[Age = age_parents - (j_star - 1)]
                end
        @assert (sum(υ[ϕ = :NI, Age = j, ϕ_next = :I]) ≈ 1.) & (sum(υ[ϕ = :NI, Age = j, ϕ_next = :NI]) ≈ 1.) & (sum(υ[ϕ = :I, Age = j, ϕ_next = :I]) ≈ 1.) & (sum(υ[ϕ = :I, Age = j, ϕ_next = :NI]) ≈ 0.)
    end
    return υ
end

function state_transition(z_next::Symbol, z::Symbol, ϕ_next::Symbol, ϕ::Symbol, age::Int64, Params::NamedTuple)
    (; z_chain, ψ) = Params
    age_parents = age + 20
    # Get the index of the current/next idiosync shock in the transition matrix
    z_i = z_chain.state_values .== z
    next_z_i = z_chain.state_values .== z_next

    ϕ_chain = [1. 0.; 1 - ψ[age_parents + 1] ψ[age_parents + 1]]
    ϕ_i = [:I, :NI] .== ϕ
    next_ϕ_i = [:I, :NI] .== ϕ_next

    π_ = z_chain.p[z_i, next_z_i] * ϕ_chain[ϕ_i, next_ϕ_i]
    return π_[:][begin]
end

function ubound(a_past::Float64, z::Symbol, age::Int64, r::Float64, w::Float64, Params::NamedTuple, Policy::NamedTuple)
    (; z_chain, a_min) = Params
    idx = get_state_value_index(z_chain, z)
    q = get_dispo_income(w, Params, Policy)

    # Case when you save everything, no consumption only savings
    c =  a_min
    ub = (1 + r) * a_past + q[age, idx] - c
    ub = ifelse(ub <= a_min, a_min, ub)
    return [ub]
end

function c_transition(a_past::Float64, a::Vector{Float64}, z::Symbol, age::Int64, r::Float64, w::Float64, Params::NamedTuple, Policy::NamedTuple)
    (; z_chain) = Params
    idx = get_state_value_index(z_chain, z)
    q = get_dispo_income(w, Params, Policy)

    c = (1 + r) * a_past + q[age, idx] - a[begin] 
    return c
end

function obj(V::AxisArray{Float64, 2}, a::Vector{Float64}, a_past::Float64, z::Symbol, age::Int64, r::Float64, w::Float64, Params::NamedTuple, Policy::NamedTuple)
    (; ψ, β, a_vals, β, u, U, J, j_star) = Params
    c = c_transition(a_past, a, z, age, r, w, Params, Policy)

    if age == J
        VF = u(c) + β * U(a[begin])
    else
        # # Extract the value function
        v = V[Age = age + 1 - (j_star-1)]
        # Interpolate the value function on the asset for the following state
        v_new = CubicSplineInterpolation(a_vals, v, extrapolation_bc = Line())(a[begin])
        VF = u(c) + β * ψ[age + 1] * v_new + β * (1 - ψ[age + 1]) * U(a[begin])
    end
    return VF
end

function obj(V::AxisArray{Float64, 4}, a::Vector{Float64}, a_past::Float64, z::Symbol, ϕ::Symbol, age::Int64, r::Float64, w::Float64, υ::AxisArray, Params::NamedTuple, Policy::NamedTuple)
    (; ψ, β, z_chain, a_vals, β, u, U) = Params
    c = c_transition(a_past, a, z, age, r, w, Params, Policy)
    # Initialise the expectation
    Ev_new = 0 
    for z_next ∈ z_chain.state_values
        for ϕ_next ∈ [:I, :NI]
            π_ = state_transition(z_next, z, ϕ_next, ϕ, age, Params)
            # # Extract the value function a skill given for the next shock
            v = V[Age = age + 1, Z = z_next, ϕ = ϕ_next]
            # Compute the average size of bequest that one may get
            b = dot(υ[ϕ = ϕ, Age = age, ϕ_next = ϕ_next], a_vals)
            # Interpolate the value function on the asset for the following state AND add bequest
            v_new = CubicSplineInterpolation(a_vals, v, extrapolation_bc = Line())(a[begin] + b)
            # Compute the expectation
            Ev_new += v_new * π_
        end
    end
    # print(a[begin], "\n")
    VF = u(c) + β* ψ[age + 1] * Ev_new + β * (1 - ψ[age + 1]) * U(a[begin])
    return VF
end

function get_dr(r::Float64, w::Float64, υ::AxisArray, Params::NamedTuple, Policy::NamedTuple)
    (; β, z_chain, z_size, a_size, a_vals, a_min, β, j_star, J) = Params
    
    # Active population
    V_a = AxisArray(zeros(a_size, z_size, 2, j_star-1);
                    a = 1:a_size,
                    Z = (z_chain.state_values),
                    ϕ = [:I, :NI],
                    Age = 1:j_star-1
            )
    A_a = AxisArray(zeros(a_size, z_size, 2, j_star-1);
                        a = 1:a_size,
                        Z = (z_chain.state_values),
                        ϕ = [:I, :NI],
                        Age = 1:j_star-1
                )
    C_a = AxisArray(zeros(a_size, z_size, 2, j_star-1);
        a = 1:a_size,
        Z = (z_chain.state_values),
        ϕ = [:I, :NI],
        Age = 1:j_star-1
    )

    # Retired
    V_r = AxisArray(zeros(a_size, J-(j_star-1));
    a = 1:a_size,
    Age = j_star:J
    )
    A_r = AxisArray(zeros(a_size, J-(j_star-1));
        a = 1:a_size,
        Age = j_star:J
    )
    C_r = AxisArray(zeros(a_size, J-(j_star-1));
    a = 1:a_size,
    Age = j_star:J
    )

    # Loop over ages recursively
    for j ∈ ProgressBar(J:-1:1)
        # Loop over past assets
        for (a_past_i, a_past) ∈ enumerate(a_vals)
            if j >= j_star
                age_i = j- (j_star-1)
                ## Optimization for retired 
                # Specify lower bound for optimisation
                lb = [a_min]
                # Specify upper bound for optimisation
                ub = ubound(a_past, :U, j, r, w, Params, Policy)
                # Specify the initial value in the middle of the range
                init = (lb + ub)/2
                Sol = optimize(x -> -obj(V_r, x, a_past, :U, j, r, w, Params, Policy), lb, ub, init)
                @assert Optim.converged(Sol)
                a = Optim.minimizer(Sol)

                # Deduce optimal value function, consumption and asset
                V_r[Age = age_i, a = a_past_i] = -1 * Optim.minimum(Sol)
                C_r[Age = age_i, a = a_past_i] = c_transition(a_past, a, :U, j, r, w, Params, Policy)
                A_r[Age = age_i, a = a_past_i] = a[begin]
            else    
                ## Optimization for workers 
                # Loop over idiosync shock
                for z ∈ z_chain.state_values
                    for ϕ ∈ [:I, :NI]
                        # Specify lower bound for optimisation
                        # lb = ifelse((z == :U) & (a_past_i == 1), [a_min] .+ 1e-5,  [a_min])
                        lb = [a_min] .+ 1e-5
                        # Specify upper bound for optimisation
                        ub = ubound(a_past, z, j, r, w, Params, Policy)
                        # Specify the initial value in the middle of the range
                        init = (lb + ub)/2
                        # Optimization
                        if j == j_star - 1
                            Sol = optimize(x -> -obj(V_r, x, a_past, z, j, r, w, Params, Policy), lb, ub, init)

                        else
                            Sol = optimize(x -> -obj(V_a, x, a_past, z, ϕ, j, r, w, υ, Params, Policy), lb, ub, init)
                        end
                        @assert Optim.converged(Sol)
                        a = Optim.minimizer(Sol)

                        # Deduce optimal value function, consumption and asset
                        V_a[Age = j, Z = z, a = a_past_i, ϕ = ϕ] = -1 * Optim.minimum(Sol)
                        C_a[Age = j, Z = z, a = a_past_i, ϕ = ϕ] = c_transition(a_past, a, z, j, r, w, Params, Policy)
                        A_a[Age = j, Z = z, a = a_past_i, ϕ = ϕ] = a[begin]
                    end
                end
            end
        end
    end
    return (Act = (V = V_a, C = C_a, A = A_a), Ret = (V = V_r, C = C_r, A = A_r))
end

υ = bequest_distrib(λ, HHs)
r = 0.04
w = 2.
dr = get_dr(r, w, υ, HHs, Policies)
