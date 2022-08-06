using Heterogenous_Agent_Model, QuantEcon, LaTeXStrings, Parameters, Plots, Serialization, StatsPlots, AxisArrays
# TODO : Equilibrer les taxes pour l'équilibre générale
# TODO : Simplifier les AxisArray pour les retraités
# TODO : Rajouter le taux de croissance de la pop
# TODO : Calculer le welfare pour policy

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
                    j_star = retirement_age - age_start_work + 1,
                    J = death_age - age_start_work + 1,
                    ψ = import_aging_prob(age_start_work, death_age), # Probabilities to survive
                    μ = get_pop_distrib(ψ), # Population distribution
                    ϵ = get_efficiency(age_start_work, retirement_age - 1), #Efficiency index
                    b = get_soc_sec_benefit(ϵ, h, w, j_star, J, Policies),
                    W = get_wages(ϵ, h, w, Policies),
                    q = get_dispo_income(W, b, j_star, Policies),
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
r = 0.02
w = r_to_w(r, Firm)
τ_ssc = 0.1
τ_u = 0.1
Policies = Policy()
HHs = Households();
L0 = 0.94 * HHs.h * sum(HHs.μ[1:HHs.j_star-1] .* HHs.ϵ)


K = 3.
B = 1.

x = solve_equilibrium(
    K, 
    B,
    η_tol_K=1e-3,
    η_tol_B=1e-3
)


get_r(x.K, 0.94 * HHs.h * sum(HHs.μ[1:HHs.j_star-1] .* HHs.ϵ), Firm)


CC = get_aggregate_C(x.λ,  x.dr.C)
II = get_aggregate_I(x.λ,  x.dr.A, Firm, HHs)
YY = get_aggregate_Y(x.λ,  x.dr.A, Firm, HHs)

YY - CC - II

dot(x.dr.C, x.λ)

function check_GE(dr::NamedTuple, λ::AxisArray{Float64, 3},)
    # Consumption
    # Investment
    # Output

end



x.K
x.B

plot(x.dr.V[Age = 65])

bar(HHs.a_vals, sum(x.λ[Age = 60] / HHs.μ[60], dims=2))
bar!(HHs.a_vals,sum(x.λ[Age = 54] / HHs.μ[54], dims=2))
bar!(HHs.a_vals, sum(x.λ[Age = 50] / HHs.μ[50], dims=2))
bar!(HHs.a_vals, sum(x.λ[Age = 44] / HHs.μ[44], dims=2))

bar(HHs.a_vals, sum(x.λ[Age = 60], dims=2))
bar!(HHs.a_vals,sum(x.λ[Age = 54], dims=2))
bar!(HHs.a_vals, sum(x.λ[Age = 50], dims=2))
bar!(HHs.a_vals, sum(x.λ[Age = 44], dims=2))


r = get_r(x.K, 0.94 * HHs.h * sum(HHs.μ[1:HHs.j_star-1] .* HHs.ϵ), Firm)
w = r_to_w(r, Firm)

HHs = Households()
KK = x.K
LL = get_aggregate_L(x.λ, HHs)
CC = dot(x.dr.C, x.λ)
YY = Firm.Ω * KK^Firm.α * LL^(1-Firm.α)
A_past = similar(x.dr.A)
A_past[Age = 2:65] = x.dr.A[Age = 1:64]
A_past[Age = 1] = zeros(HHs.a_size, 2)
KK_past = get_aggregate_K(x.λ, A_past)


CC + KK - YY - (1 - Firm.δ) * KK_past




dr = get_dr(pm)
sim = simulate_OLG(dr.A, pm, N=10000);
λ = get_ergodic_distribution(sim, pm)
get_ergodic_distribution(sim, pm, PopScaled = true)



bar(λ[Age = 60, Z = :E])
bar!(λ[Age = 54, Z = :E])
bar!(λ[Age = 50, Z = :E])
bar!(λ[Age = 44, Z = :E])
bar!(λ[Age = 1, Z = :E])


pl = plot()
xlabel!(L"Age")
ylabel!(L"Assets")
for n=1:3000
    plot!(pl, sim.A[N = n] , label=nothing, color=:red, alpha=0.1)
end
pl

λ.mean(axis=(2,3))
bar(dropdims(sum(dropdims(sum(λ, dims=3), dims=3), dims=2), dims=2))
bar!(dropdims(sum(dropdims(sum(λ2, dims=3), dims=3), dims=2), dims=2))

A[Age = 1]

plot(dr.A[Age = 1])
plot(dr.C[Age = 45])
plot(dr.V[Age = 45])
