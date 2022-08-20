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

Firm = Firms();
HHs = Households();

Results = Dict()
for θ ∈ range(0,1,11)
    Results[θ] = solve_equilibrium(
                                    4., 
                                    0.94 * HHs.h * sum(HHs.μ[1:HHs.j_star-1] .* HHs.ϵ),
                                    deserialize("data/distrib_bequest.dat"),
                                    Firm,
                                    HHs,
                                    Policy(θ = θ)
                                    )
end


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

