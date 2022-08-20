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
                    Σ = 2., # Constant relative risk aversion (asset utility)
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
                    U = Σ == 1 ? a -> log(a) : a -> 15. * (a^(1 - Σ)) / (1 - Σ),
)

Firms = @with_kw ( 
    α = 0.36,
    Ω = 1.2082, #1.3193,
    δ = 0.08,
)


using LinearAlgebra, Statistics, Interpolations, Optim, ProgressBars, Printf, QuantEcon, CSV, NLsolve, AxisArrays, Distributions, DataStructures, Parameters

function bequest_distrib(λ::NamedTuple, Params::NamedTuple)
    (; a_size, j_star) = Params
    υ = AxisArray(zeros(a_size, 2, 2, j_star-1);
        b = 1:a_size,
        ϕ_next = [:I, :NI],
        ϕ = [:I, :NI],
        Age = 1:j_star-1
    )

    for j ∈ 1:j_star-1
        age_parents = j + 20
        # Already inherited so for sure no more bequest and state doesn't change 
        υ[ϕ = :I, Age = j, ϕ_next = :I, b = 1] = 1.
        # Not inherited yet 
            ## Parents are still alive
            υ[ϕ = :NI, Age = j, ϕ_next = :NI, b = 1] = 1. #ψ[age_parents]
            ## Parents just dead
                if age_parents < j_star
                    ### Parents are not retired
                    υ[ϕ = :NI, Age = j, ϕ_next = :I] = sum(sum(λ.λ_a[Age = age_parents], dims = 3), dims=2) #(1 - ψ[age_parents]) .* sum(λ.λ_a[Age = age_parents], dims = 2)
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

function ubound(a_past::Float64, z::Symbol, age::Int64, r::Float64, q::AxisArray{Float64, 2}, Params::NamedTuple)
    (; a_min) = Params

    # Case when you save everything, no consumption only savings
    c =  a_min
    ub = (1 + r) * a_past + q[Age = age, Z=z] - c
    ub = ifelse(ub <= a_min, a_min, ub)
    return [ub]
end

function obj(V::AxisArray{Float64, 2}, a::Vector{Float64}, a_past::Float64, z::Symbol, age::Int64, r::Float64, q::AxisArray{Float64, 2}, Params::NamedTuple)
    (; ψ, β, a_vals, β, u, U, J, j_star) = Params
    c = (1+r) * a_past - a[begin] + q[Age = age, Z=z]

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

function obj(V::AxisArray{Float64, 4}, a::Vector{Float64}, a_past::Float64, z::Symbol, ϕ::Symbol, age::Int64, r::Float64, q::AxisArray{Float64, 2}, υ::AxisArray, Params::NamedTuple)
    (; ψ, β, z_chain, a_vals, β, u, U) = Params
    c = (1+r) * a_past - a[begin] + q[Age = age, Z=z]
    # Initialise the expectation
    Ev_new = 0 
    for z_next ∈ z_chain.state_values
        for ϕ_next ∈ [:I, :NI]
            π_ = state_transition(z_next, z, ϕ_next, ϕ, age, Params)
            if π_ == 0.
                break
            end
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

function get_dr(r::Float64, q::AxisArray{Float64, 2}, υ::AxisArray, Params::NamedTuple)
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
                ub = ubound(a_past, :U, j, r, q, Params)
                # Specify the initial value in the middle of the range
                init = (lb + ub)/2
                Sol = optimize(x -> -obj(V_r, x, a_past, :U, j, r, q, Params), lb, ub, init)
                @assert Optim.converged(Sol)
                a = Optim.minimizer(Sol)

                # Deduce optimal value function, consumption and asset
                V_r[Age = age_i, a = a_past_i] = -1 * Optim.minimum(Sol)
                C_r[Age = age_i, a = a_past_i] = (1+r) * a_past - a[begin] + q[Age = j, Z=:U]
                A_r[Age = age_i, a = a_past_i] = a[begin]
            else    
                ## Optimization for workers 
                # Loop over idiosync shock
                for z ∈ z_chain.state_values
                    for ϕ ∈ [:I, :NI]
                        # Specify lower bound for optimisation
                        # lb = ifelse((z == :U) & (a_past_i == 1), [a_min] .+ 1e-5,  [a_min])
                        lb = [a_min] #.+ 1e-5
                        # Specify upper bound for optimisation
                        ub = ubound(a_past, z, j, r, q, Params)
                        # Specify the initial value in the middle of the range
                        init = (lb + ub)/2
                        # Optimization
                        if j == j_star - 1
                            Sol = optimize(x -> -obj(V_r, x, a_past, z, j, r, q, Params), lb, ub, init)

                        else
                            Sol = optimize(x -> -obj(V_a, x, a_past, z, ϕ, j, r, q, υ, Params), lb, ub, init)
                        end
                        @assert Optim.converged(Sol)
                        a = Optim.minimizer(Sol)

                        # Deduce optimal value function, consumption and asset
                        V_a[Age = j, Z = z, a = a_past_i, ϕ = ϕ] = -1 * Optim.minimum(Sol)
                        C_a[Age = j, Z = z, a = a_past_i, ϕ = ϕ] = (1+r) * a_past - a[begin] + q[Age = j, Z=z]
                        A_a[Age = j, Z = z, a = a_past_i, ϕ = ϕ] = a[begin]
                    end
                end
            end
        end
    end
    return (Act = (V = V_a, C = C_a, A = A_a), Ret = (V = V_r, C = C_r, A = A_r))
end

function get_distribution(dr::NamedTuple, Params::NamedTuple; PopScaled::Bool = false)
    (;a_size, z_size, j_star, z_chain, a_vals, J, μ) = Params

    λ_a = AxisArray(zeros(a_size, 2, z_size, j_star-1);
    a = 1:a_size,
    ϕ = [:I, :NI],
    Z = (z_chain.state_values),
    Age = 1:j_star-1
    )

    λ_r = AxisArray(zeros(a_size, J-(j_star-1));
    a = 1:a_size,
    Age = j_star:J
    )

    # Initial wealth distribution
    λ_a[Age=1, a=1, ϕ = :NI] = [0.06 0.94] 

    for j ∈ 2:j_star-1
        for a_past ∈ 1:a_size
            for z_past = z_chain.state_values
                for ϕ_past = [:I, :NI]
                    for z = z_chain.state_values
                        for ϕ = [:I, :NI]
                            res = get_weight(dr.Act.A[Age=j, Z=z, ϕ=ϕ, a=a_past], a_vals)
                            λ_a[Age = j, Z=z, ϕ=ϕ, a = res.lb] += λ_a[Age=j-1, a=a_past, Z=z_past, ϕ=ϕ_past] * state_transition(z, z_past, ϕ, ϕ_past, j, HHs) * (1 - res.w0)
                            λ_a[Age = j, Z=z, ϕ=ϕ, a = res.ub] += λ_a[Age=j-1, a=a_past, Z=z_past, ϕ=ϕ_past] * state_transition(z, z_past, ϕ, ϕ_past, j, HHs) * res.w0
                        end
                    end
                end
            end
        end
        @assert sum(λ_a[Age=j]) ≈ 1.
    end

    for j ∈ j_star
        for a_past ∈ 1:a_size
            for z_past = z_chain.state_values
                for ϕ_past = [:I, :NI]
                    res = get_weight(dr.Ret.A[Age= j - (j_star-1), a=a_past], a_vals)
                    λ_r[Age = j - (j_star-1), a = res.lb] += λ_a[Age=j-1, a=a_past, Z=z_past, ϕ=ϕ_past] * (1 - res.w0)
                    λ_r[Age = j - (j_star-1), a = res.ub] += λ_a[Age=j-1, a=a_past, Z=z_past, ϕ=ϕ_past] * res.w0
                end
            end
        end
        @assert sum(λ_r[Age=j - (j_star-1)]) ≈ 1.
    end

    for j ∈ j_star+1:J
        for a_past ∈ 1:a_size
            res = get_weight(dr.Ret.A[Age= j - (j_star-1), a=a_past], a_vals)
            λ_r[Age = j - (j_star-1), a = res.lb] += λ_r[Age=j-j_star, a=a_past] * (1 - res.w0)
            λ_r[Age = j - (j_star-1), a = res.ub] += λ_r[Age=j-j_star, a=a_past] * res.w0
        end
        @assert sum(λ_r[Age=j - (j_star-1)]) ≈ 1.
    end

    if PopScaled
        for j ∈ 1:j_star-1
            λ_a[Age = j] *= μ[j]
        end
        for j ∈ j_star:J
            λ_r[Age = j - (j_star-1)] *= μ[j]
        end
        @assert (sum(λ_a) + sum(λ_r)) ≈ 1.
    else
        @assert (sum(λ_a) + sum(λ_r))/ J ≈ 1.
    end
    return (λ_a=λ_a, λ_r=λ_r)
end

function get_HH_distributions(υ0::AxisArray{Float64, 4}, r::Float64, q::AxisArray{Float64, 2}, Params::NamedTuple; maxit=300, η_tol = 1e-5)  
    η0 = 1.0
    iter = ProgressBar(1:maxit)
    for n in iter

        dr = get_dr(r, q, υ0, Params)
        λ = get_distribution(dr, Params)
        υ1 = bequest_distrib(λ, Params)

        η = maximum(υ1 .- υ0)

        λ_ = η/η0
        η0 = η

        if η<η_tol
            λ_scaled = get_distribution(dr, Params, PopScaled = true)
            return (υ1=υ1, dr=dr, λ=λ, λ_scaled=λ_scaled)
        end

        υ0 = υ1
        set_postfix(iter, η=@sprintf("%.8f", η), λ=@sprintf("%.8f", λ_))

    end
end

function solve_equilibrium(K0::Float64, L0::Float64, υ0::AxisArray{Float64, 4}, Firms, Households, Policy; maxit=300, η_tol = 1e-3, α=0.5)
    
    # Initial values
    Firm = Firms;
    Policies = Policy
    HHs = Households

    η0 = 1.0
    iter = ProgressBar(1:maxit)
    for n in iter
        ## Firm 
        r = get_r(K0, L0, Firm)
        w = r_to_w(r, Firm)

        # Households 
        q = get_dispo_income(w, HHs, Policies)
        res = get_HH_distributions(υ0, r, q, HHs)

        # Aggregation
        K1 = dot(res.λ_scaled.λ_a, res.dr.Act.A) + dot(res.λ_scaled.λ_r, res.dr.Ret.A)

        η = abs(K1 - K0) 

        λ = η/η0

        η0 = η

        if η<η_tol
            println("\n Algorithm stopped after iteration ", n, "\n")
            return (res=res, K=K1, r=r, w=w)
        end

        K0 = α * K0 + (1 - α) * K1
        υ0 = res.υ1

        set_postfix(iter, η=@sprintf("%.4f", η), λ=@sprintf("%.4f", λ), K=@sprintf("%.4f", K1), r=@sprintf("%.4f", r), w=@sprintf("%.4f", w))
    end
end

Firm = Firms();
HHs = Households();

K = 4.
L = 0.94 * HHs.h * sum(HHs.μ[1:HHs.j_star-1] .* HHs.ϵ)
υ0 = deserialize("data/distrib_bequest.dat")

Policies = Policy(θ = 0.3)
x = solve_equilibrium(K, L, υ0, Firm, HHs, Policies)


# Consumption profiles
plot(reshape(
    hcat(dropdims(dropdims(sum(sum(sum(x.res.λ.λ_a .* x.res.dr.Act.C, dims=1), dims=2), dims=3), dims=3), dims=2),
    sum(x.res.λ.λ_r .* x.res.dr.Ret.C, dims=1)),
         :,1)
     )


# Wealth profiles
plot(reshape(
    hcat(dropdims(dropdims(sum(sum(sum(x.res.λ.λ_a .* x.res.dr.Act.A, dims=1), dims=2), dims=3), dims=3), dims=2),
    sum(x.res.λ.λ_r .* x.res.dr.Ret.A, dims=1)),
         :,1)
     )

# Overall distribution of wealth
bar(HHs.a_vals,
    dropdims(dropdims(sum(sum(sum(x.res.λ_scaled.λ_a, dims=2), dims=3), dims=4) .+ sum(x.res.λ_scaled.λ_r, dims=2), dims=2), dims=3)
    )


# Distribution of wealth for specific cohorts
bar(HHs.a_vals,
    dropdims(sum(sum(x.res.λ.λ_a[Age = 44], dims=2), dims=3), dims=3),
    label= L"44 ans"
    )

bar!(HHs.a_vals,
    x.res.λ.λ_r[Age = 6],
    label= L"50 ans"
    )

bar!(HHs.a_vals,
    x.res.λ.λ_r[Age = 10],
    label= L"54 ans"
    )

bar!(HHs.a_vals,
    x.res.λ.λ_r[Age = 16],
    label= L"60 ans"
    )

