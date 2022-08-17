using Heterogenous_Agent_Model, QuantEcon, LaTeXStrings, Parameters, Plots, Serialization, StatsPlots, AxisArrays
# TODO : Checker la simulation des agents
# TODO : Equilibrer les taxes pour l'équilibre générale
# TODO : Simplifier les AxisArray pour les retraités

Policy = @with_kw (
                    ξ = 0.4,
                    θ = 0.3,
                    τ_ssc = 0.067061,
                    τ_u = 0.06/0.94 * ξ
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
Policies = Policy()
HHs = Households();


function bequest_distrib(b_next, ϕ_next, age, ϕ)
    if ϕ == :Inherited
        # Already inherited so for sure no more bequest and state doesn't change 
        if (ϕ_next == :Inherited) & (b_next = 0.)
            return 1.
        else
            return 0.
        end

    else  #if agent did not inherited yet 

        if ϕ_next == :NotInherited & (b_next = 0.)
            # equal the probability that parents are still alive
            return ψ[age + 15]
        elseif ϕ_next == :NotInherited & (b_next != 0.)
            # Can't have a positive bequest if did not inherit
            return 0.
        elseif ϕ_next == Inherited
            return measure of bequest which depends on age and  (on va utiliser le résultat du modèle précédent. Save les resultats et réutiliser balec)




    end
    

end


(; β, z_chain, z_size, a_size, a_vals, a_min, β, j_star, J) = HHs

# Active population
V_a = AxisArray(zeros(a_size, z_size, 2, j_star-1);
                a = 1:a_size,
                Z = (z_chain.state_values),
                ϕ = (:Inherited, :NotInherited),
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
                ub = ubound(a_past, :U, j, r, w, B, Params, Policy)
                # Specify the initial value in the middle of the range
                init = (lb + ub)/2
                Sol = optimize(x -> -obj(V_r, x, a_past, :U, j, r, w, B, Params, Policy), lb, ub, init)
                @assert Optim.converged(Sol)
                a = Optim.minimizer(Sol)

                # Deduce optimal value function, consumption and asset
                V_r[Age = age_i, a = a_past_i] = -1 * Optim.minimum(Sol)
                C_r[Age = age_i, a = a_past_i] = c_transition(a_past, a, :U, j, r, w, B, Params, Policy)
                A_r[Age = age_i, a = a_past_i] = a[begin]
            else    
                ## Optimization for workers 
                # Loop over idiosync shock
                for z ∈ z_chain.state_values
                    # Specify lower bound for optimisation
                    lb = [a_min]        
                    # Specify upper bound for optimisation
                    ub = ubound(a_past, z, j, r, w, B, Params, Policy)
                    # Specify the initial value in the middle of the range
                    init = (lb + ub)/2
                    # Optimization
                    if j == j_star - 1
                        Sol = optimize(x -> -obj(V_r, x, a_past, z, j, r, w, B, Params, Policy), lb, ub, init)

                    else
                        Sol = optimize(x -> -obj(V_a, x, a_past, z, j, r, w, B, Params, Policy), lb, ub, init)
                    end
                    @assert Optim.converged(Sol)
                    a = Optim.minimizer(Sol)

                    # Deduce optimal value function, consumption and asset
                    V_a[Age = j, Z = z, a = a_past_i] = -1 * Optim.minimum(Sol)
                    C_a[Age = j, Z = z, a = a_past_i] = c_transition(a_past, a, z, j, r, w, B, Params, Policy)
                    A_a[Age = j, Z = z, a = a_past_i] = a[begin]
                end
            end
        end
    end
    return (Act = (V = V_a, C = C_a, A = A_a), Ret = (V = V_r, C = C_r, A = A_r))
end
