using Heterogenous_Agent_Model, QuantEcon, LaTeXStrings, Parameters, Plots, Serialization, StatsPlots, AxisArrays

Policy = @with_kw (
                    ξ = 0.4,
                    θ = 0.3,
                    τ_ssc = 0.1,
                    τ_u = 0.1
)    

Households = @with_kw ( 
                    r = 0.04, # interest rate
                    w = 1.,
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
                    z_chain = MarkovChain([0.9 0.1;
                                            0.1 0.9], 
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
    A = 1.3193,
    δ = 0.08,
)

Policies = Policy()
HHs = Households();
Firm = Firms();

dr = get_dr(pm)
sim = simulate_OLG(dr.A, pm, N=10000);
λ = get_ergodic_distribution(sim, pm)
get_ergodic_distribution(sim, pm, PopScaled = true)


# Initial values
Policies = Policy();
Firm = Firms();
HHs = Households();
L = 0.94 * HHs.h * sum(HHs.μ[1:HHs.j_star-1] .* HHs.ϵ)
K = 3.
maxit = 100
η_tol = 1e-4

η0 = 1.0
iter = ProgressBar(1:maxit)
for n in iter
    ## Firm 
    r = get_r(K, L, Firm)
    w = get_w(K, L, Firm)

    # Households 
    HHs = Households(r = r, w = w);
    dr = get_dr(HHs)
    sim = simulate_OLG(dr.A, HHs, N=1000);
    λ = get_ergodic_distribution(sim, HHs, PopScaled = true)

    # Aggregation
    K1 = get_aggregate_K(λ,  dr.A)
    L1 = get_aggregate_L(λ, HHs)

    η = maximum(abs, K1 - K)

    λ = η/η0
    η0 = η
    if verbose
        println(n, " : ", η, " : ", λ)
    end

    K = K1
    L = L1

    if η<η_tol
        println("\n Algorithm stopped after iteration ", n, ", with μ = ", λ, "\n")
        return (λ=λ, dr=dr, sim=sim, K=K, L=L )
    end
    set_postfix(iter, η=@sprintf("%.8f", η), λ=@sprintf("%.8f", λ))
end


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
