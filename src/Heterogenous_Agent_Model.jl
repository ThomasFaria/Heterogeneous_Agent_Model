module Heterogenous_Agent_Model

using LinearAlgebra, Statistics, Interpolations, Optim, ProgressBars, Printf, QuantEcon, CSV, NLsolve, AxisArrays, Distributions, DataStructures, Parameters


function import_aging_prob(age_min::Int64, age_max::Int64)
    data = CSV.read("data/LifeTables.csv", NamedTuple)
    ψ = data.Prob2survive
    # After a certain age death is certain
    ψ[age_max+2:end] .= 0
    # TODO: Est ce qu'il faut pas mettre +2 ici? à verifier
    return ψ[age_min+1:age_max+2]
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
    # Find first share that assures a sum equal to 1
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


function get_soc_sec_benefit(w::Float64, Households::NamedTuple, Policy::NamedTuple)
    (; θ) = Policy
    (; ϵ, h, j_star, J) = Households

    W = w * h * ϵ
    b  = zeros(J)
    b[j_star:end] .= θ * mean(W)

    return b
end
export get_soc_sec_benefit

function get_dispo_income(w::Float64, Households::NamedTuple, Policy::NamedTuple)
    (; ξ, τ_ssc, τ_u) = Policy
    (; ϵ, h, j_star) = Households

    b = get_soc_sec_benefit(w, Households, Policy)

    q = zeros(size(b, 1), 2)

    q[begin:j_star-1,1] .= w * h * ξ
    q[begin:j_star-1,2] = (1 - τ_ssc - τ_u) * w * h * ϵ
    q[j_star:end,:] .= b[j_star:end]
    return q
end
export get_dispo_income

function state_transition(z_next::Symbol, z::Symbol, Params::NamedTuple)
    (; z_chain) = Params

    # Get the index of the current/next idiosync shock in the transition matrix
    z_i = z_chain.state_values .== z
    next_z_i = z_chain.state_values .== z_next

    π = z_chain.p[z_i, next_z_i]
    return π[:][begin]
end
export state_transition

function get_state_value_index(mc::MarkovChain, value)
    idx = findall(mc.state_values .== value)
    return idx[begin]
end 
export get_state_value_index

function c_transition(a_past::Float64, a::Vector{Float64}, z::Symbol, age::Int64, r::Float64, w::Float64, B::Float64, Params::NamedTuple, Policy::NamedTuple)
    (; z_chain) = Params
    idx = get_state_value_index(z_chain, z)
    q = get_dispo_income(w, Params, Policy)

    c = (1 + r) * a_past + q[age, idx] + B - a[begin] 
    return c
end
export c_transition

function ubound(a_past::Float64, z::Symbol, age::Int64, r::Float64, w::Float64, B::Float64, Params::NamedTuple, Policy::NamedTuple)
    (; z_chain, a_min) = Params
    idx = get_state_value_index(z_chain, z)
    q = get_dispo_income(w, Params, Policy)

    # Case when you save everything, no consumption only savings
    c =  a_min
    ub = (1 + r) * a_past + q[age, idx] + B - c
    ub = ifelse(ub <= a_min, a_min, ub)

    return [ub]
end
export ubound

function obj(V::AxisArray{Float64, 3}, a::Vector{Float64}, a_past::Float64, z::Symbol, age::Int64, r::Float64, w::Float64, B::Float64, Params::NamedTuple, Policy::NamedTuple)
    (; ψ, β, z_chain, a_vals, β, u, J, j_star) = Params
    c = c_transition(a_past, a, z, age, r, w, B, Params, Policy)

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
            π = state_transition(z_next, z, Params)
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
export obj

function get_dr(r::Float64, w::Float64, B::Float64, Params::NamedTuple, Policy::NamedTuple)
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
    for j ∈ ProgressBar(J:-1:1)
        # Loop over past assets
        for a_i ∈ 1:a_size
            a = a_vals[a_i]
            # Loop over idiosync shock
            for z ∈ z_chain.state_values
                # Specify lower bound for optimisation
                lb = [a_min]        
                # Specify upper bound for optimisation
                ub = ubound(a, z, j, r, w, B, Params, Policy)
                # Specify the initial value in the middle of the range
                init = (lb + ub)/2
                # Optimization
                Sol = optimize(x -> -obj(V, x, a, z, j, r, w, B, Params, Policy), lb, ub, init)
                a_new = Optim.minimizer(Sol)

                # Deduce optimal value function, consumption and asset
                V[Age = j, Z = z, a = a_i] = obj(V, a_new, a, z, j, r, w, B, Params, Policy)
                C[Age = j, Z = z, a = a_i] = c_transition(a, a_new, z, j, r, w, B, Params, Policy)
                A[Age = j, Z = z, a = a_i] = a_new[begin]

            end
        end
    end
    return (V = V, C = C, A = A)
end
export get_dr

function simulate_model(dr::AxisArray{Float64, 3}, r::Float64, w::Float64, B::Float64, Params::NamedTuple, Policy::NamedTuple; Initial_Z = 0.06, N=1)
    (; J, ψ, z_chain, a_vals, a_min) = Params

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
                # In age 1, agent holds 0 asset and so doesn't consume
                a = a_min
                c = a_min
                # Draw the employment state
                prob2Unemp = Binomial(1, Initial_Z)
                IsUnem = Bool(rand(prob2Unemp, 1)[begin])
                z = ifelse(IsUnem, :U, :E)

                # Saving the results    
                A[Age = j, N = n] = a
                C[Age = j, N = n] = c
                Z[Age = j, N = n] = z
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
                A[Age = j, N = n] = CubicSplineInterpolation(a_vals, 
                                                            dr[Z = Z[Age = j, N = n], Age = j], 
                                                            extrapolation_bc = Line())(A[Age = j-1, N = n])

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
    (; a_size, z_size, J, z_chain, a_vals, μ) = Params
    
    λ = AxisArray(zeros(a_size, z_size, J);
    a = 1:a_size,
    Z = (z_chain.state_values),
    Age = 1:J
    )

    for j ∈ 1:J
        N = size(filter(!isnan,sim.A[Age = j]), 1)
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
                    w = [(key - lb)./(ub - lb) for key ∈ keys(vals)]
                    λ0 = [vals[key]/N for key ∈ keys(vals)]

                    λ[Age = j, Z = z, a = i] += sum((1 .- w) .* λ0)
                    if lb == a_vals[end]
                        # If assets exceeding the grid, put everything on the last value
                        λ[Age = j, Z = z, a = i] += sum(w .* λ0)
                    else
                        λ[Age = j, Z = z, a = i+1] += sum(w .* λ0)
                    end
                end
            end
        end
        # println(" By age : ", round(sum(λ[Age = j]), digits = 8), "\n")
        # @assert sum(λ[Age = j]) ≈ 1.
        if PopScaled
            λ[Age = j] *= μ[j]
        end
    end
    if PopScaled
        # println(" Total : ", round(sum(λ), digits = 8), "\n")
        # @assert sum(λ) ≈ 1.
    end
    return λ
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

function get_SSC_rate(λ::AxisArray{Float64, 3}, w::Float64, Params::NamedTuple, Policy::NamedTuple)
    (; j_star, J, h, ϵ) = Params

    b = get_soc_sec_benefit(w, Params, Policy)

    SSC = sum(λ[Age = j_star:J]) * b[j_star]
    tax_base = sum(λ[Age = 1:j_star-1, Z = :E] * ϵ ) * w * h
    return SSC/tax_base
end
export get_SSC_rate

function get_U_benefit_rate(λ::AxisArray{Float64, 3}, w::Float64, Params::NamedTuple, Policy::NamedTuple)
    (; j_star, J, h, ϵ) = Params
    (; ξ) = Policy
    U_benefit = sum(λ[Age = j_star:J]) * ξ * w * h
    tax_base = sum(λ[Age = 1:j_star-1, Z = :E] * ϵ ) * w * h
    return U_benefit/tax_base
end
export get_U_benefit_rate

function get_aggregate_K(λ::AxisArray{Float64, 3},  A::AxisArray{Float64, 3}, Params::NamedTuple)
    (; a_size, z_size, J, a_min) = Params
    A_past = similar(A)
    A_past[Age = 2:J] = A[Age = 1:J-1]
    A_past[Age = 1] = ones(a_size, z_size) * a_min
    return dot(A_past, λ)
end
export get_aggregate_K

function get_aggregate_L(λ::AxisArray{Float64, 3}, Households::NamedTuple)
    (; ϵ, h, j_star) = Households
    return sum(λ[Z = :E, Age = 1:j_star - 1] * ϵ * h)
end
export get_aggregate_L

function get_aggregate_B(λ::AxisArray{Float64, 3},  A::AxisArray{Float64, 3}, Params::NamedTuple)
    (; ψ, J) = Params
    B = similar(A)
    for i ∈ 1:J
        B[Age = i] =  (1 - ψ[i+1]) * A[Age = i]
    end
    return dot(B, λ)
end
export get_aggregate_B

function get_aggregate_C(λ::AxisArray{Float64, 3},  C::AxisArray{Float64, 3})
    return dot(C, λ)
end
export get_aggregate_C

function get_aggregate_I(λ::AxisArray{Float64, 3},  A::AxisArray{Float64, 3}, Firm::NamedTuple, Params::NamedTuple)
    (; δ) = Firm
    K = dot(λ, A)
    K_past = get_aggregate_K(λ, A, Params)
    return K - (1 - δ) * K_past
end
export get_aggregate_I

function get_aggregate_Y(λ::AxisArray{Float64, 3},  A::AxisArray{Float64, 3}, Firm::NamedTuple, Params::NamedTuple)
    (; Ω, α) = Firm
    K = get_aggregate_K(λ, A, Params)
    L = get_aggregate_L(λ, Params)
    return Ω * K^α * L^(1 - α)
end
export get_aggregate_Y

function check_GE(dr::NamedTuple, λ::AxisArray{Float64, 3}, Households::NamedTuple, Firms::NamedTuple)
    # Consumption
    C = get_aggregate_C(λ,  dr.C)
    # Investment
    I = get_aggregate_I(λ,  dr.A, Firms, Households)
    # Output
    Y = get_aggregate_Y(λ,  dr.A, Firms, Households)
    return Y - (C + I)
end
export check_GE

function solve_equilibrium(K0::Float64, L0::Float64,  B0::Float64, Firms, Households, Policy; N=2000, maxit=300, η_tol_K=1e-3, η_tol_B=1e-3, α_K=0.4, α_B=0.4)
    
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
        dr = get_dr(r, w, B0, HHs, Policies)
        sim = simulate_model(dr.A, r, w, B0, HHs, Policies, N=N);
        λ = get_ergodic_distribution(sim, HHs, PopScaled = true)

        # Aggregation
        K1 = get_aggregate_K(λ, dr.A, HHs)
        B1 = get_aggregate_B(λ, dr.A, HHs)
        L1 = get_aggregate_L(λ, HHs)

        # Policies
        τ_ssc = get_SSC_rate(λ, w, HHs, Policies)
        τ_u = get_U_benefit_rate(λ, w, HHs, Policies)
        

        η_K = abs(K1 - K0) 
        η_B = abs(B1 - B0) 

        λ_K = η_K/η0_K
        λ_B = η_B/η0_B

        η0_K = η_K
        η0_B = η_B

        if (η_K<η_tol_K) & (η_B<η_tol_B)
            println("\n Algorithm stopped after iteration ", n, "\n")
            check_GE(dr, λ, HHs, Firm) > 0.001  && @warn "Markets are not clearing"
            return (λ=λ, dr=dr, sim=sim, K=K1, B = B1, r=r, w=w, HH = HHs, τ_ssc=τ_ssc, τ_u=τ_u)
        end
        println("Convergence : ", check_GE(dr, λ, HHs, Firm))
        K0 = α_K * K0 + (1 - α_K) * K1
        B0 = α_B * B0 + (1 - α_B) * B1
        L0 = L1

        set_postfix(iter, K=@sprintf("%.4f", K1), B=@sprintf("%.4f", B1), CV=@sprintf("%.4f", τ_ssc), η_K=@sprintf("%.4f", η_K), λ_K=@sprintf("%.4f", λ_K))
    end
end
export solve_equilibrium

end
