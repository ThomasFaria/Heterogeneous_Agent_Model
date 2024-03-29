module Heterogenous_Agent_Model

using LinearAlgebra, Statistics, Interpolations, Optim, ProgressBars, Printf, QuantEcon, CSV, NLsolve, AxisArrays, Distributions, DataStructures, Plots,         LaTeXStrings, Parameters


function import_aging_prob(age_min::Int64, age_max::Int64)
    data = CSV.read("data/LifeTables.csv", NamedTuple)
    ψ = data.Prob2survive
    # After a certain age death is certain
    ψ[age_max+2:end] .= 0
    # TODO: Est ce qu'il faut pas mettre +2 ici? à verifier
    return ψ[age_min+1:age_max+2]
end
export import_aging_prob

function pop_distrib(μ_1::Vector{Float64}, ψ::Vector{Float64}, ρ::Float64, find_root::Bool)
    μ = zeros(size(ψ))
    μ[begin] = μ_1[1]

    for (index, value) in enumerate(ψ[2:end])
        μ[index+1] = value/(1+ρ) * μ[index]
    end

    if find_root
        return sum(μ) - 1
    else
        return μ
    end
end

function get_pop_distrib(ψ::Vector{Float64}, ρ::Float64)
    # Find first share that assures a sum equal to 1
    sol = nlsolve(x -> pop_distrib(x, ψ, ρ, true), [0.02])
    μ = pop_distrib(sol.zero, ψ, ρ, false)
    return μ
end
export get_pop_distrib

function get_efficiency(age_min::Int64, age_max::Int64)
    ϵ =  [exp(-1.46 + 0.070731*j - 0.00075*j^2) for j in age_min:age_max]
    return ϵ
end
export get_efficiency

function get_soc_sec_benefit(w::Float64, Households::NamedTuple, Policy::NamedTuple)
    (; θ) = Policy
    (; ϵ, h) = Households

    b = θ * mean(w * h * ϵ)

    return b
end
export get_soc_sec_benefit

function get_dispo_income(w::Float64, Households::NamedTuple, Policy::NamedTuple)
    (; ξ, τ_ssc, τ_u) = Policy
    (; ϵ, h, j_star, J, z_size, z_chain) = Households

    b = get_soc_sec_benefit(w, Households, Policy)
    w_e = w * h * ϵ

    q = AxisArray(zeros(J, z_size);
        Age = 1:J,
        Z = (z_chain.state_values)
    )

    q[Age = begin:j_star-1, Z = :U] = ξ * w_e
    q[Age = begin:j_star-1, Z = :E] = (1 - τ_ssc - τ_u) * w_e
    q[Age = j_star:J] = b * ones(J-(j_star-1), z_size)
    return q
end
export get_dispo_income

function state_transition(z_next::Symbol, z::Symbol, Params::NamedTuple)
    (; z_chain) = Params

    # Get the index of the current/next idiosync shock in the transition matrix
    z_i = z_chain.state_values .== z
    next_z_i = z_chain.state_values .== z_next

    π_ = z_chain.p[z_i, next_z_i]
    return π_[:][begin]
end
export state_transition

function get_state_value_index(mc::MarkovChain, value)
    idx = findall(mc.state_values .== value)
    return idx[begin]
end 
export get_state_value_index

function c_transition(a_past::Float64, a::Vector{Float64}, z::Symbol, age::Int64, r::Float64, q::AxisArray{Float64, 2}, B::Float64)

    c = (1 + r) * a_past + q[Age = age, Z=z] + B - a[begin] 
    return c
end
export c_transition

function ubound(a_past::Float64, z::Symbol, age::Int64, r::Float64, q::AxisArray{Float64, 2}, B::Float64, Params::NamedTuple)
    (; a_min) = Params

    # Case when you save everything, no consumption only savings
    c =  a_min
    ub = (1 + r) * a_past + q[Age = age, Z=z] + B - c
    ub = ifelse(ub <= a_min, a_min, ub)

    return [ub]
end
export ubound

function obj(V::AxisArray{Float64, 3}, a::Vector{Float64}, a_past::Float64, z::Symbol, age::Int64, r::Float64, q::AxisArray{Float64, 2}, B::Float64, Params::NamedTuple)
    (; ψ, β, z_chain, a_vals, β, u) = Params
    c = (1+r) * a_past - a[begin] + q[Age = age, Z=z] + B

    # Initialise the expectation
    Ev_new = 0
    for z_next ∈ z_chain.state_values
        π_ = state_transition(z_next, z, Params)
        # # Extract the value function a skill given for the next shock
        v = V[Age = age + 1, Z = z_next]
        # Interpolate the value function on the asset for the following state
        v_new = CubicSplineInterpolation(a_vals, v, extrapolation_bc = Line())(a[begin])
        # Compute the expectation
        Ev_new += v_new * π_
    end
    VF = u(c) + β* ψ[age + 1] * Ev_new
    return VF
end

function obj(V::AxisArray{Float64, 2}, a::Vector{Float64}, a_past::Float64, z::Symbol, age::Int64, r::Float64, q::AxisArray{Float64, 2}, B::Float64, Params::NamedTuple)
    (; ψ, β, a_vals, β, u, J, j_star) = Params

    c = (1+r) * a_past - a[begin] + q[Age = age, Z=z] + B
    if age == J
        VF = u(c) 
    else
        # # Extract the value function
        v = V[Age = age + 1 - (j_star-1)]
        # Interpolate the value function on the asset for the following state
        v_new = CubicSplineInterpolation(a_vals, v, extrapolation_bc = Line())(a[begin])
        VF = u(c) + β * ψ[age + 1] * v_new
    end
    return VF
end
export obj

function get_dr(r::Float64, q::AxisArray{Float64, 2}, B::Float64, Params::NamedTuple)
    (; β, z_chain, z_size, a_size, a_vals, a_min, β, j_star, J) = Params
    
    # Active population
    V_a = AxisArray(zeros(a_size, z_size, j_star-1);
                    a = 1:a_size,
                    Z = (z_chain.state_values),
                    Age = 1:j_star-1
            )
    A_a = AxisArray(zeros(a_size, z_size, j_star-1);
                        a = 1:a_size,
                        Z = (z_chain.state_values),
                        Age = 1:j_star-1
                )
    C_a = AxisArray(zeros(a_size, z_size,  j_star-1);
        a = 1:a_size,
        Z = (z_chain.state_values),
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
                ub = ubound(a_past, :U, j, r, q, B, Params)
                # Specify the initial value in the middle of the range
                init = (lb + ub)/2
                Sol = optimize(x -> -obj(V_r, x, a_past, :U, j, r, q, B, Params), lb, ub, init)
                @assert Optim.converged(Sol)
                a = Optim.minimizer(Sol)

                # Deduce optimal value function, consumption and asset
                V_r[Age = age_i, a = a_past_i] = -1 * Optim.minimum(Sol)
                A_r[Age = age_i, a = a_past_i] = a[begin]
                # C_r[Age = age_i, a = a_past_i] = (1+r) * a_past - A_r[Age = age_i, a = a_past_i] + q[Age = j, Z=:U] + B
            else    
                ## Optimization for workers 
                # Loop over idiosync shock
                for z ∈ z_chain.state_values
                    # Specify lower bound for optimisation
                    lb = [a_min]        
                    # Specify upper bound for optimisation
                    ub = ubound(a_past, z, j, r, q, B, Params)
                    # Specify the initial value in the middle of the range
                    init = (lb + ub)/2
                    # Optimization
                    if j == j_star - 1
                        Sol = optimize(x -> -obj(V_r, x, a_past, z, j, r, q, B, Params), lb, ub, init)

                    else
                        Sol = optimize(x -> -obj(V_a, x, a_past, z, j, r, q, B, Params), lb, ub, init)
                    end
                    @assert Optim.converged(Sol)
                    a = Optim.minimizer(Sol)

                    # Deduce optimal value function, consumption and asset
                    V_a[Age = j, Z = z, a = a_past_i] = -1 * Optim.minimum(Sol)
                    A_a[Age = j, Z = z, a = a_past_i] = a[begin]
                    # C_a[Age = j, Z = z, a = a_past_i] = (1+r) * a_past - a[begin] + q[Age = j, Z=z] + B
                    # @assert A_a[Age = j, Z = z, a = a_past_i] == (1+r) * a_past + q[Age = j, Z=z] + B - C_a[Age = j, Z = z, a = a_past_i]

                end
            end
        end
    end

    for j ∈ 1:j_star-1
        for z ∈ z_chain.state_values
            C_a[Age = j, Z=z] = (1+r) * a_vals .- A_a[Age = j, Z=z] .+ q[Age = j, Z=z] .+ B
        end
    end

    for j ∈ j_star:J
       C_r[Age = j - (j_star-1)] = (1+r) * a_vals .- A_r[Age = j - (j_star-1)] .+ q[Age = j, Z=:U] .+ B
    end


    return (Act = (V = V_a, C = C_a, A = A_a), Ret = (V = V_r, C = C_r, A = A_r))
end
export get_dr

function simulate_model(dr::NamedTuple, r::Float64, w::Float64, B::Float64, Params::NamedTuple, Policy::NamedTuple; Initial_Z = 0.074, N=1)
    (; J, ψ, z_chain, a_vals, a_min, j_star) = Params

    A = AxisArray(fill(NaN, (J, N));
    Age = 1:J,
    N = 1:N
    );

    C = AxisArray(fill(NaN, (J, N));
    Age = 1:J,
    N = 1:N
    );

    Z = AxisArray(fill(:DEAD, (J, N));
    Age = 1:J,
    N = 1:N
    );

    for n ∈ ProgressBar(1:N)
        for j ∈ 1:J

            if j == 1
                # In age 0, agent holds 0 asset
                a_0 = a_min

                # Draw the employment state
                prob2Unemp = Binomial(1, Initial_Z)
                IsUnem = Bool(rand(prob2Unemp, 1)[begin])
                z = ifelse(IsUnem, :U, :E)

                # Saving the results
                Z[Age = j, N = n] = z    
                A[Age = j, N = n] = a_min
                C[Age = j, N = n] = c_transition(a_0, 
                                                [A[Age = j, N = n]], 
                                                Z[Age = j, N = n], 
                                                j,
                                                r, 
                                                w, 
                                                B,
                                                Params, 
                                                Policy
                                                )
            else
                # Draw the survival outcome
                prob2survive = Binomial(1, ψ[j])
                Survived = Bool(rand(prob2survive, 1)[begin])
                if !Survived
                    # the agent died at the beginning of the period
                    break
                end
                # Draw the employment state based on previous employment state
                Z[Age = j, N = n] = simulate(z_chain, 2, 
                                                init = get_state_value_index(z_chain, Z[Age = j-1, N = n]))[end]

                # Extrapolate the next asset based on the optimal decision rule
                if j <= j_star-1
                    a = CubicSplineInterpolation(a_vals, 
                                                dr.Act.A[Z = Z[Age = j, N = n], Age = j], 
                                                extrapolation_bc = Line())(A[Age = j-1, N = n])
                    A[Age = j, N = n] = ifelse(a <= a_min, a_min, a)
                else
                    a = CubicSplineInterpolation(a_vals, 
                                                dr.Ret.A[Age = j- (j_star-1)], 
                                                extrapolation_bc = Line())(A[Age = j-1, N = n])
                    A[Age = j, N = n] = ifelse(a <= a_min, a_min, a)
                end
                # Compute the current consumption
                C[Age = j, N = n] = c_transition(A[Age = j-1, N = n], 
                                                    [A[Age = j, N = n]], 
                                                    Z[Age = j, N = n], 
                                                    j,
                                                    r, 
                                                    w, 
                                                    B,
                                                    Params, 
                                                    Policy)
            end
        end
    end

    return (A = A, Z = Z, C = C)

end
export simulate_model

function get_ergodic_distribution(sim::NamedTuple, Params::NamedTuple; PopScaled::Bool = false)
    (; a_size, z_size, J, z_chain, a_vals, μ, j_star) = Params
    
    λ_a = AxisArray(zeros(a_size, z_size, j_star-1);
    a = 1:a_size,
    Z = (z_chain.state_values),
    Age = 1:j_star-1
    )
    λ_r = AxisArray(zeros(a_size, J-(j_star-1));
    a = 1:a_size,
    Age = j_star:J
    )

    for j ∈ 1:J
        N = size(filter(!isnan,sim.A[Age = j]), 1)
        if j<= j_star-1
            # Workers
            for z ∈ z_chain.state_values
                for (i, lb) ∈ enumerate(a_vals)
                    if lb == a_vals[end]
                        ub = Inf
                    else
                        ub = a_vals[i + 1]
                    end

                    # We collect all assets in a certain interval of the grid
                    idx = ((sim.A[Age = j][sim.Z[Age = j] .== z] .>= lb)) .&  (sim.A[Age = j][sim.Z[Age = j] .== z] .< ub) 
                    vals = counter(sim.A[Age = j][sim.Z[Age = j] .== z][idx])

                    # We check that this set is not empty
                    if !isempty(vals)
                        w0 = [(key - lb)./(ub - lb) for key ∈ keys(vals)]
                        λ0 = [vals[key]/N for key ∈ keys(vals)]

                        λ_a[Age = j, Z = z, a = i] += sum((1 .- w0) .* λ0)
                        if lb == a_vals[end]
                            # If assets exceeding the grid, put everything on the last value
                            λ_a[Age = j, Z = z, a = i] += sum(w0 .* λ0)
                        else
                            λ_a[Age = j, Z = z, a = i+1] += sum(w0 .* λ0)
                        end
                    end
                end
            end
            @assert sum(λ_a[Age = j]) ≈ 1.
            if PopScaled
                λ_a[Age = j] *= μ[j]
            end        
        else
            #Retired
            for (i, lb) ∈ enumerate(a_vals)
                if lb == a_vals[end]
                    ub = Inf
                else
                    ub = a_vals[i + 1]
                end
                age_i = j- (j_star-1)
                # We collect all assets in a certain interval of the grid
                idx = ((sim.A[Age = j].>= lb)) .&  (sim.A[Age = j] .< ub) 
                vals = counter(sim.A[Age = j][idx])

                # We check that this set is not empty
                if !isempty(vals)
                    w0 = [(key - lb)./(ub - lb) for key ∈ keys(vals)]
                    λ0 = [vals[key]/N for key ∈ keys(vals)]

                    λ_r[Age = age_i, a = i] += sum((1 .- w0) .* λ0)
                    if lb == a_vals[end]
                        # If assets exceeding the grid, put everything on the last value
                        λ_r[Age = age_i, a = i] += sum(w0 .* λ0)
                    else
                        λ_r[Age = age_i, a = i+1] += sum(w0 .* λ0)
                    end
                end
            end
            @assert sum(λ_r[Age = j- (j_star-1)]) ≈ 1.
            if PopScaled
                λ_r[Age =  j- (j_star-1)] *= μ[j]
            end
        end
    end
    if PopScaled
        # println(" Total : ", round(sum(λ), digits = 8), "\n")
        @assert sum(λ_a)+sum(λ_r) ≈ 1.
    end
    return (λ_a=λ_a, λ_r=λ_r)
end
export get_ergodic_distribution

function get_r(K::Float64, L::Float64, Firm::NamedTuple)
    (;α, Ω, δ) = Firm
    return Ω * α * (L/K)^(1-α) - δ
end
export get_r

function r_to_w(r::Float64, Firm::NamedTuple)
    (;α, Ω, δ) = Firm
    return Ω * (1 - α) * (Ω * α / (r + δ)) ^ (α / (1 - α))
end
export r_to_w

function get_SSC_rate(λ::NamedTuple, w::Float64, Params::NamedTuple, Policy::NamedTuple)
    (; h, ϵ) = Params

    b = get_soc_sec_benefit(w, Params, Policy)

    SSC = sum(λ.λ_r) * b
    tax_base = sum(λ.λ_a[Z=:E]* ϵ) * w * h
    return SSC/tax_base
end
export get_SSC_rate

function get_U_benefit_rate(λ::NamedTuple, w::Float64, Params::NamedTuple, Policy::NamedTuple)
    (; h, ϵ) = Params
    (; ξ) = Policy
    U_benefit = sum(λ.λ_a[Z=:U]* ϵ) * ξ * w * h
    tax_base = sum(λ.λ_a[Z=:E]* ϵ) * w * h
    return U_benefit/tax_base
end
export get_U_benefit_rate

function get_aggregate_B(λ::NamedTuple,  dr::NamedTuple, Params::NamedTuple)
    (; j_star, ψ, z_size, a_size, J) = Params

    ψ_reshaped_a_next = reshape(repeat(ψ[2:j_star], inner = (a_size * z_size)), (a_size, z_size, j_star-1))
    ψ_reshaped_r_next = reshape(repeat(ψ[j_star+1:J+1], inner = (a_size)), (a_size, J - (j_star-1)))

    return dot(λ.λ_a .* (1 .- ψ_reshaped_a_next), dr.Act.A) + dot(λ.λ_r .* (1 .- ψ_reshaped_r_next), dr.Ret.A)
end
export get_aggregate_B

function get_aggregate_K(λ::NamedTuple,  dr::NamedTuple)
    return dot(λ.λ_a, dr.Act.A) + dot(λ.λ_r, dr.Ret.A)
end
export get_aggregate_K

function get_aggregate_C(λ::NamedTuple,  dr::NamedTuple)
    return dot(λ.λ_a, dr.Act.C) + dot(λ.λ_r, dr.Ret.C)
end
export get_aggregate_C

function get_aggregate_Y(λ::NamedTuple,  dr::NamedTuple, L::Float64, Firm::NamedTuple)
    (; Ω, α) = Firm
    K = get_aggregate_K(λ, dr)
    return Ω * K^α * L^(1 - α)
end
export get_aggregate_Y

function get_aggregate_Welfare(λ::NamedTuple,  dr::NamedTuple, Params::NamedTuple)
    (; J, j_star, β, ψ, u) = Params
       
    W_a = sum(reshape(sum(sum(u.(dr.Act.C) .* λ.λ_a, dims=1), dims=2), :,1) .* [β^(j-1) * prod(ψ[2:j+1]) for j ∈ 1:j_star-1])

    W_r = sum(reshape(sum(u.(dr.Ret.C) .* λ.λ_r, dims=1), :,1) .* [β^(j-1) * prod(ψ[2:j+1]) for j ∈ j_star:J])
    
    return W_a + W_r
end
export get_aggregate_Welfare

function check_GE(dr::NamedTuple, λ::NamedTuple, L::Float64, Firms::NamedTuple)
    (; δ) = Firms
    # Consumption
    C = get_aggregate_C(λ, dr)
    # Capital
    K = get_aggregate_K(λ, dr)
    # Output
    Y = get_aggregate_Y(λ,  dr, L, Firms)
    return Y - (C + δ * K)
end
export check_GE

function get_weight(val, range)
    idx = argmin(abs.(range .- val))
    if val >= range[end]
        w0 = 0
        idx_lb = size(range,1)
        idx_ub = size(range,1)
    else
        if range[idx] <= val
            lb = range[idx]
            ub = range[idx+1]
        else range[idx] >= val
            lb = range[idx-1]
            ub = range[idx]
            idx -= 1
        end

        w0 = (val - lb)/(ub - lb) 
        idx_lb  = idx
        idx_ub  = idx+1
    end
    return (w0=w0, lb=idx_lb, ub=idx_ub)
end
export get_weight

function get_distribution_(dr::NamedTuple, Params::NamedTuple; PopScaled::Bool = false)
    (;a_size, z_size, j_star, z_chain, a_vals, J, μ) = Params

    λ_a = AxisArray(zeros(a_size, z_size, j_star-1);
    a = 1:a_size,
    Z = (z_chain.state_values),
    Age = 1:j_star-1
    )

    λ_r = AxisArray(zeros(a_size, J-(j_star-1));
    a = 1:a_size,
    Age = j_star:J
    )

    # Initial wealth distribution
    λ_a[Age=1, a=1] = [0.074 1-0.074] 

    for j ∈ 2:j_star-1
        for a_past ∈ 1:a_size
            for z_past = z_chain.state_values
                for z = z_chain.state_values
                    res = get_weight(dr.Act.A[Age=j, Z=z, a=a_past], a_vals)
                    λ_a[Age = j, Z=z, a = res.lb] += λ_a[Age=j-1, a=a_past, Z=z_past] * state_transition(z, z_past, Params) * (1 - res.w0)
                    λ_a[Age = j, Z=z, a = res.ub] += λ_a[Age=j-1, a=a_past, Z=z_past] * state_transition(z, z_past, Params) * res.w0
                end
            end
        end
        @assert sum(λ_a[Age=j]) ≈ 1.
    end

    for j ∈ j_star
        for a_past ∈ 1:a_size
            for z_past = z_chain.state_values
                res = get_weight(dr.Ret.A[Age= j - (j_star-1), a=a_past], a_vals)
                λ_r[Age = j - (j_star-1), a = res.lb] += λ_a[Age=j-1, a=a_past, Z=z_past] * (1 - res.w0)
                λ_r[Age = j - (j_star-1), a = res.ub] += λ_a[Age=j-1, a=a_past, Z=z_past] * res.w0
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
export get_distribution

function solve_equilibrium(K0::Float64, L0::Float64,  B0::Float64, Firms, Households, Policy; N=10000, maxit=300, η_tol_K=1e-3, η_tol_B=1e-3, α_K=0.5, α_B=0.5)
    
    # Initial values
    Firm = Firms;
    Policies = Policy
    HHs = Households

    η0_K = 1.0
    η0_B = 1.0
    iter = ProgressBar(1:maxit)
    for n in iter
        ## Firm 
        r = get_r(K0, L0, Firm)
        w = r_to_w(r, Firm)

        # Households 
        q = get_dispo_income(w, HHs, Policies)
        dr = get_dr(r, q, B0, HHs)
        # sim = simulate_model(dr, r, w, B0, HHs, Policies, N=N);
        # λ = get_ergodic_distribution(sim, HHs, PopScaled = true)
        λ = get_distribution_(dr, HHs, PopScaled = true)

        # Aggregation
        K1 = get_aggregate_K(λ, dr)
        B1 = get_aggregate_B(λ, dr, HHs)


        η_K = abs(K1 - K0) 
        η_B = abs(B1 - B0) 

        λ_K = η_K/η0_K

        η0_K = η_K
        η0_B = η_B

        if (η_K<η_tol_K) & (η_B<η_tol_B)
            println("\n Algorithm stopped after iteration ", n, "\n")
            # check_GE(dr, λ, HHs, Firm) > 0.001  && @warn "Markets are not clearing"
            λ_scaled = λ
            λ = get_distribution_(dr, HHs, PopScaled = false)
            C = get_aggregate_C(λ_scaled, dr)
            Y = get_aggregate_Y(λ_scaled, dr, L0, Firm)
            W = get_aggregate_Welfare(λ, dr, HHs)
            q = get_dispo_income(w, HHs, Policy)
            return (λ=λ, λ_scaled=λ_scaled, dr=dr, q=q, K=K1, B = B1, C = C, Y = Y, W = W, r=r, w=w, Policy=Policies, Households=HHs)
        end

        K0 = α_K * K0 + (1 - α_K) * K1
        B0 = α_B * B0 + (1 - α_B) * B1

        Y = get_aggregate_Y(λ, dr, L0, Firm)
        set_postfix(iter, K=@sprintf("%.4f", K1), B=@sprintf("%.4f", B1), Y=@sprintf("%.4f", Y), r=@sprintf("%.4f", r), w=@sprintf("%.4f", w), η_K=@sprintf("%.4f", η_K), λ_K=@sprintf("%.4f", λ_K))
    end
end
export solve_equilibrium

############################################################################################################
################################# SECOND MODEL WITH BEQUEST MOTIVE #########################################
############################################################################################################

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
export bequest_distrib

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
    λ_a[Age=1, a=1, ϕ = :NI] = [0.074 1-0.074] 

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
export get_HH_distributions

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
            C = get_aggregate_C(res.λ_scaled, res.dr)
            Y = get_aggregate_Y(res.λ_scaled, res.dr, L0, Firm)
            W = get_aggregate_Welfare(res.λ, res.dr, HHs)

            return (λ=res.λ, λ_scaled=res.λ_scaled, dr=res.dr, K=K1, C = C, Y = Y, W = W, r=r, w=w, Policy=Policies)
            return (res=res, K=K1, r=r, w=w)
        end

        K0 = α * K0 + (1 - α) * K1
        υ0 = res.υ1

        set_postfix(iter, η=@sprintf("%.4f", η), λ=@sprintf("%.4f", λ), K=@sprintf("%.4f", K1), r=@sprintf("%.4f", r), w=@sprintf("%.4f", w))
    end
end

############################################################################################################
############################################### PLOTTING ###################################################
############################################################################################################

function plot_consumption_profiles(λ_a::AxisArray{Float64,3}, C_a::AxisArray{Float64,3}, λ_r::AxisArray{Float64,2}, C_r::AxisArray{Float64,2}, w::Float64, Params::NamedTuple, Policy::NamedTuple)
    (; J, j_star, h, ϵ) = Params
    (; ξ, θ) = Policy

    C_a = reshape(sum(sum(λ_a .* C_a, dims=2), dims=1), :, 1)
    C_r = reshape(sum(λ_r .* C_r, dims=1), :, 1)

    w_e = w * h * ϵ
    w_u = ξ .* w_e 
    b = θ * mean(w_e)

    Y_a = sum(dropdims(sum(λ_a, dims=1), dims=1)' .* hcat(w_u, w_e), dims=2)
    Y_r = [b for i in j_star:J]

    p = plot((1:J) .+ 20
        , vcat(C_a, C_r)
        , label="Consumption", legend=:bottomright
        )
    
    plot!(p
    , (1:J) .+ 20
    , vcat(Y_a, Y_r)
    , label="Pre-tax income", legend=:bottomright)

    xlabel!(L"Age")

    return p
end

export plot_consumption_profiles

function plot_wealth_profiles(λ_a::AxisArray{Float64,3}, A_a::AxisArray{Float64,3}, λ_r::AxisArray{Float64,2}, A_r::AxisArray{Float64,2}, Params::NamedTuple)
    (; J) = Params
    A_a = reshape(sum(sum(λ_a .* A_a, dims=2), dims=1), :, 1)
    A_r = reshape(sum(λ_r .* A_r, dims=1), :, 1)

    p = plot((1:J) .+ 20
        , vcat(A_a, A_r)
        , label=nothing
        )
    xlabel!(L"Age")
    ylabel!(L"Asset")

    return p
end
export plot_wealth_profiles

function plot_wealth_profiles_multiple(Results::Dict, Policies::Vector{Float64})
    p = plot()

    for θ ∈ Policies
        (; J) = Results[θ].Households
        A_a = reshape(sum(sum(Results[θ].λ.λ_a .* Results[θ].dr.Act.A, dims=2), dims=1), :, 1)
        A_r = reshape(sum(Results[θ].λ.λ_r .* Results[θ].dr.Ret.A, dims=1), :, 1)

        plot!(p
            , (1:J) .+ 20
            , vcat(A_a, A_r)
            , label= @sprintf("θ = %.2f", θ)
            )
    end

    xlabel!(L"Age")
    ylabel!(L"Asset")

    return p
end
export plot_wealth_profiles_multiple

function plot_wealth_distrib(λ_scaled_a::AxisArray{Float64,3}, λ_scaled_r::AxisArray{Float64,2}, Params::NamedTuple)
    (; a_vals) = Params
    distrib = sum(sum(λ_scaled_a, dims=2), dims=3) .+ sum(λ_scaled_r, dims=2)

    p = bar(a_vals
        , reshape(distrib, :,1)
        , label=nothing
        )
    xlabel!(L"Asset")
    return p
end
export plot_wealth_distrib

function plot_wealth_by_age(λ_a::AxisArray{Float64,3}, λ_r::AxisArray{Float64,2}, ages::Vector{Int64},  Params::NamedTuple)
    (; a_vals, j_star) = Params

    p = plot()
    for j ∈ ages 
        true_age = j+20
        if j < j_star
            bar!(p
            , a_vals
            , sum(λ_a[Age = j], dims=2)
            , label= @sprintf("%.0f years", true_age)
            )
        else
            bar!(p
            , a_vals
            , λ_r[Age = j - (j_star-1)]
            , label= @sprintf("%.0f years", true_age)
            )

        end
    end

    xlabel!(L"Asset")
    return p
end
export plot_wealth_by_age

end