using Heterogenous_Agent_Model, QuantEcon, LaTeXStrings, Parameters, Plots, Serialization, StatsPlots, AxisArrays
# TODO : Equilibrer les taxes pour l'équilibre générale

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
                    # U = Σ == 1 ? a -> log(a) : a -> (a^(1 - Σ)) / (1 - Σ),
)
Firms = @with_kw ( 
    α = 0.36,
    Ω = 1.3175, #1.3193,
    δ = 0.08,
)

# Initial values
Firm = Firms();
HHs = Households();

K = 4.
B = 0.5
L = 0.94 * HHs.h * sum(HHs.μ[1:HHs.j_star-1] .* HHs.ϵ)

Policies = Policy(θ = 0.1)
x1 = solve_equilibrium(
    K, 
    L,
    B,
    Firm,
    HHs,
    Policies, 
    η_tol_K=1e-2,
    η_tol_B=1e-2
)


Policies = Policy(θ = 0.2)
x2 = solve_equilibrium(
    K, 
    L,
    B,
    Firm,
    HHs,
    Policies, 
    η_tol_K=1e-3,
    η_tol_B=1e-3
)

Policies = Policy(θ = 0.3)
x3 = solve_equilibrium(
    K, 
    L,
    B,
    Firm,
    HHs,
    Policies, 
    η_tol_K=1e-3,
    η_tol_B=1e-3
)

Policies = Policy(θ = 0.4)
x4 = solve_equilibrium(
    K, 
    L,
    B,
    Firm,
    HHs,
    Policies, 
    η_tol_K=1e-3,
    η_tol_B=1e-3
)


Policies = Policy(θ = 0.5)
x5 = solve_equilibrium(
    K, 
    L,
    B,
    Firm,
    HHs,
    Policies, 
    η_tol_K=1e-3,
    η_tol_B=1e-3
)


Policies = Policy(θ = 0.6)
x6 = solve_equilibrium(
    K, 
    L,
    B,
    Firm,
    HHs,
    Policies, 
    η_tol_K=1e-3,
    η_tol_B=1e-3
)



Policies = Policy(θ = 0.7)
x7 = solve_equilibrium(
    K, 
    L,
    B,
    Firm,
    HHs,
    Policies, 
    η_tol_K=1e-3,
    η_tol_B=1e-3
)


Policies = Policy(θ = 0.8)
x8 = solve_equilibrium(
    K, 
    L,
    B,
    Firm,
    HHs,
    Policies, 
    η_tol_K=1e-3,
    η_tol_B=1e-3
)


Policies = Policy(θ = 0.9)
x9 = solve_equilibrium(
    K, 
    L,
    B,
    Firm,
    HHs,
    Policies, 
    η_tol_K=1e-3,
    η_tol_B=1e-3
)



Policies = Policy(θ = 1.)
x10 = solve_equilibrium(
    K, 
    L,
    B,
    Firm,
    HHs,
    Policies, 
    η_tol_K=1e-3,
    η_tol_B=1e-3
)

## PLOTS
λ_ = get_distribution(x3.dr, HHs, PopScaled = false)
serialize("data/distrib_03.dat", λ_)
# Consumption profiles
plot(reshape(
    hcat(dropdims(sum(sum(λ_.λ_a .* x.dr.Act.C, dims=2), dims=1), dims=1), sum(λ_.λ_r .* x.dr.Ret.C, dims=1)),
     :,1)
     )

# Wealth profiles
plot(reshape(
    hcat(dropdims(sum(sum(λ_.λ_a .* x.dr.Act.A, dims=2), dims=1), dims=1), sum(λ_.λ_r .* x.dr.Ret.A, dims=1)),
     :,1)
     )

# Overall distribution of wealth
bar(HHs.a_vals,
    sum(dropdims(sum(x.λ.λ_a, dims=2), dims=2), dims=2) .+ sum(x.λ.λ_r, dims=2),
    )


# Distribution of wealth for specific cohorts
bar(HHs.a_vals,
    sum(x3.λ.λ_a[Age = 44], dims=2),
    label= L"44 ans"
    )

bar!(HHs.a_vals,
    x.λ.λ_r[Age = 6],
    label= L"50 ans"
    )

bar!(HHs.a_vals,
    x.λ.λ_r[Age = 10],
    label= L"54 ans"
    )

bar!(HHs.a_vals,
    x.λ.λ_r[Age = 16],
    label= L"60 ans"
    )

bar(HHs.a_vals, sum(x.λ[Age = 60] / HHs.μ[60], dims=2))
bar!(HHs.a_vals,sum(x.λ[Age = 54] / HHs.μ[54], dims=2))
bar!(HHs.a_vals, sum(x.λ[Age = 50] / HHs.μ[50], dims=2))
bar!(HHs.a_vals, sum(x.λ[Age = 44] / HHs.μ[44], dims=2))

