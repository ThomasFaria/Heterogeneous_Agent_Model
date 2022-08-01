using Heterogenous_Agent_Model, QuantEcon, LaTeXStrings, Parameters, Plots, Serialization, StatsPlots, AxisArrays

Policy = @with_kw (
                    ξ = 0.4,
                    θ = 0.3,
                    τ_ssc = 0.1,
                    τ_u = 0.1
)    

Params = @with_kw ( 
                    death_age = 85,
                    retirement_age = 65,
                    age_start_work = 20,
                    h = 0.45,
                    w = 1.,
                    j_star = retirement_age - age_start_work,
                    J = death_age - age_start_work,
                    ψ = import_aging_prob(age_start_work, death_age), # Probabilities to survive
                    μ = get_pop_distrib(ψ), # Population distribution
                    ϵ = get_efficiency(age_start_work, retirement_age - 1), #Efficiency index
                    b = get_soc_sec_benefit(ϵ, h, w, j_star, J, Policies),
                    W = get_wages(ϵ, h, w, Policies),
                    q = get_dispo_income(W, b, j_star, Policies),
                    r = 0.04, # interest rate
                    γ = 2., # Constant relative risk aversion (consumption utility)
                    Σ = 1., # Constant relative risk aversion (asset utility)
                    β = 1.011,
                    z_chain = MarkovChain([0.9 0.1;
                                            0.1 0.9], 
                                        ["U"; "E"]),
                    # a_min = 1e-10,
                    # a_max = 10.0,
                    # a_size = 70,
                    # a_vals = range(a_min, a_max, length = a_size),
                    # z_size = length(z_chain.state_values),
                    # skill_size = length(skill_chain.state_values),
                    # age_size = length(age_chain.state_values),
                    # n = a_size * z_size * skill_size * age_size,
                    # s_vals = gridmake(a_vals, z_chain.state_values, skill_chain.state_values, age_chain.state_values),
                    # s_vals_index = gridmake(1:a_size, 1:z_size, 1:skill_size, 1:age_size),
                    # u = γ == 1 ? c -> log(c) : c -> (c^(1 - γ)) / (1 - γ),
                    # U = Σ == 1 ? a -> log(a) : a -> (a^(1 - Σ)) / (1 - Σ),
                    # Ω = Dict(zip(age_chain.state_values, [0.5; 0.75; 1.; 1.1; 0.])),
)

Firms = @with_kw ( 
    α = 0.36,
    A = 1.3193,
    δ = 0.08,
    )

Policies = Policy()
pm = Params();
Firm_pm = Firms();

function get_r(K, L, Firm_pm::NamedTuple)
    (;α, A, δ) = Firm_pm
    return A * α * (L/K)^(1-α) - δ
end

function get_w(K, L, Firm_pm::NamedTuple)
    (;α, A) = Firm_pm
    return A * (1-α) * (K/L)^α
end
