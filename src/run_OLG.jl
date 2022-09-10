using Heterogenous_Agent_Model, QuantEcon, LaTeXStrings, Parameters, Plots, Serialization, StatsPlots, AxisArrays, Printf

Policy = @with_kw (
                    ξ = 0.65,
                    θ = 0.74,
                    τ_ssc = θ * (sum(HHs.μ[HHs.j_star:end]) * mean(HHs.ϵ)) / (sum(HHs.μ[begin:HHs.j_star-1] .* HHs.ϵ)  * (1-0.074)),
                    τ_u = (0.074/(1-0.074)) * ξ
)    

Households = @with_kw ( 
                    death_age = 90,
                    retirement_age = 63,
                    age_start_work = 21,
                    h = 0.31,
                    ρ = 0.002,
                    j_star = retirement_age - age_start_work + 1,
                    J = death_age - age_start_work + 1,
                    ψ = import_aging_prob(age_start_work, death_age), # Probabilities to survive
                    μ = get_pop_distrib(ψ, ρ), # Population distribution
                    ϵ = get_efficiency(age_start_work, retirement_age - 1), #Efficiency index
                    γ = 2., # Constant relative risk aversion (consumption utility)
                    β = 1.011,
                    z_chain = MarkovChain([0.074 1-0.074;
                                           0.074 1-0.074], 
                                        [:U; :E]),
                    a_min = 1e-10,
                    a_max = 15.,
                    a_size = 100,
                    a_vals = range(a_min, a_max, length = a_size),
                    z_size = length(z_chain.state_values),
                    u = γ == 1 ? c -> log(c) : c -> (c^(1 - γ)) / (1 - γ),
)
Firms = @with_kw ( 
    α = 0.35,
    Ω = 1.75, #1.3193,
    δ = 0.07,
)

Firm = Firms();
HHs = Households();

Results = Dict()
for θ ∈ range(0,1,11)
    Results[θ] = solve_equilibrium(
                                    4., 
                                    (1-0.074) * HHs.h * sum(HHs.μ[1:HHs.j_star-1] .* HHs.ϵ),
                                    0.5,
                                    Firm,
                                    HHs,
                                    Policy(θ = θ), 
                                    η_tol_K=1e-3,
                                    η_tol_B=1e-3
                                )
    
end

Results[θ] = solve_equilibrium(
    4., 
    (1-0.074) * HHs.h * sum(HHs.μ[1:HHs.j_star-1] .* HHs.ϵ),
    0.5,
    Firm,
    HHs,
    Policy(θ = 0.74), 
    η_tol_K=1e-3,
    η_tol_B=1e-3
)

serialize("data/Results.dat", Results)


######################################################
######################################################

Results = deserialize("data/Results.dat")

## PLOTS
θ = 0.74
plot_consumption_profiles(Results[θ].λ.λ_a, Results[θ].dr.Act.C
                        , Results[θ].λ.λ_r, Results[θ].dr.Ret.C
                        , Results[θ].q
                        , Results[θ].Households)

plot_wealth_profiles(Results[θ].λ.λ_a, Results[θ].dr.Act.A
                   , Results[θ].λ.λ_r, Results[θ].dr.Ret.A
                   , Results[θ].Households)

plot_wealth_profiles_multiple(Results, [0.0,0.3,0.74, 0.9])
plot_wealth_profiles_multiple(Results, [θ])

plot_wealth_distrib(Results[θ].λ_scaled.λ_a
                  , Results[θ].λ_scaled.λ_r
                  , Results[θ].HHs)


plot_wealth_by_age(Results[θ].λ_scaled.λ_a
                 , Results[θ].λ_scaled.λ_r
                 , [24, 30, 36, 40]
                 , Results[θ].HHs)

plot_wealth_by_age(Results[θ].λ_scaled.λ_a
                 , Results[θ].λ_scaled.λ_r
                 , [44, 50, 56, 60]
                 , Results[θ].HHs)
