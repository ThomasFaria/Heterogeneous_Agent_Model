using Heterogenous_Agent_Model, QuantEcon, LaTeXStrings, Parameters, Plots, Serialization, StatsPlots, AxisArrays
# TODO : Checker la simulation des agents
# TODO : Equilibrer les taxes pour l'équilibre générale
# TODO : Simplifier les AxisArray pour les retraités

Policy = @with_kw (
                    ξ = 0.4,
                    θ = 0.3,
                    τ_ssc = 0.1,
                    τ_u = 0.1
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
Policies = Policy()
HHs = Households();

K = 3.
B = 1.
L = 0.94 * HHs.h * sum(HHs.μ[1:HHs.j_star-1] .* HHs.ϵ)

r = get_r(K, L, Firm)
w = r_to_w(r, Firm)

dr = get_dr(r, w, B, HHs, Policies)
sim = simulate_model(dr, r, w, B, HHs, Policies, N=2000)
λ = get_ergodic_distribution(sim, HHs, PopScaled = true)

K1 = get_aggregate_K(λ, dr, HHs)
B1 = get_aggregate_B(λ, dr, HHs)
L1 = get_aggregate_L(λ, HHs)

K = 0.4 * K + (1 - 0.4) * K1
B = 0.4 * B + (1 - 0.4) * B1

K = 3.
B = 1.
L = 0.94 * HHs.h * sum(HHs.μ[1:HHs.j_star-1] .* HHs.ϵ)

x = solve_equilibrium(
    K, 
    L,
    B,
    Firm,
    HHs,
    Policies, 
    η_tol_K=1e-3,
    η_tol_B=1e-3
)


CC = get_aggregate_C(x.λ,  x.dr, HHs)
II = get_aggregate_I(x.λ,  x.dr, Firm, HHs)
YY = get_aggregate_Y(x.λ,  x.dr, Firm, HHs)

YY - CC - II




## PLOTS
plot(x.dr.V[Age = 65])

bar(HHs.a_vals, sum(x.λ[Age = 60] / HHs.μ[60], dims=2))
bar!(HHs.a_vals,sum(x.λ[Age = 54] / HHs.μ[54], dims=2))
bar!(HHs.a_vals, sum(x.λ[Age = 50] / HHs.μ[50], dims=2))
bar!(HHs.a_vals, sum(x.λ[Age = 44] / HHs.μ[44], dims=2))

bar(HHs.a_vals, sum(x.λ[Age = 60], dims=2))
bar!(HHs.a_vals,sum(x.λ[Age = 54], dims=2))
bar!(HHs.a_vals, sum(x.λ[Age = 50], dims=2))
bar!(HHs.a_vals, sum(x.λ[Age = 44], dims=2))

bar(HHs.a_vals, λ[Age = 60], alpha=0.3)
bar!(HHs.a_vals, λ[Age = 54], alpha=0.3)
bar!(HHs.a_vals, λ[Age = 50], alpha=0.3)
bar!(HHs.a_vals, λ[Age = 44], alpha=0.3)
bar!(λ[Age = 1])


pl = plot()
xlabel!(L"Age")
ylabel!(L"Assets")
for n=1:2000
    plot!(pl, sim.A[N = n] , label=nothing, color=:red, alpha=0.1)
end
pl

λ.mean(axis=(2,3))
bar(dropdims(sum(dropdims(sum(λ, dims=3), dims=3), dims=2), dims=2))
bar!(dropdims(sum(dropdims(sum(λ2, dims=3), dims=3), dims=2), dims=2))

plot(dr.A[Age = 1])
plot(dr.C[Age = 45])
plot(dr.V[Age = 45])
