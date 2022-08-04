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
                                        [:U; :E]),
                    a_min = 1e-10,
                    a_max = 15.,
                    a_size = 100,
                    a_vals = range(a_min, a_max, length = a_size),
                    z_size = length(z_chain.state_values),
                    # skill_size = length(skill_chain.state_values),
                    # age_size = length(age_chain.state_values),
                    # n = a_size * z_size * skill_size * age_size,
                    # s_vals = gridmake(a_vals, z_chain.state_values, skill_chain.state_values, age_chain.state_values),
                    # s_vals_index = gridmake(1:a_size, 1:z_size, 1:skill_size, 1:age_size),
                     u = γ == 1 ? c -> log(c) : c -> (c^(1 - γ)) / (1 - γ),
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

dr = get_dr(pm)
sim = simulate_OLG(dr.A, pm, N=3000);
using DataStructures


λ = AxisArray(zeros(pm.a_size, pm.z_size, pm.J);
a = 1:pm.a_size,
Z = (pm.z_chain.state_values),
Age = 1:pm.J
)

for j ∈ 1:pm.J
    # Si on veut calculer direct avec distrib de la pop prendre N qui compte tous les ages
    N = size(filter(!isnan,sim.A[Age = j]), 1)
    for z ∈ pm.z_chain.state_values
        for (i, lb) ∈ enumerate(pm.a_vals)
            if lb == pm.a_vals[end]
                ub = Inf
            else
                ub = pm.a_vals[i + 1]
            end

            # We collect all assets in a certain interval of the grid
            idx = ((sim.A[Age = j][sim.Z[Age = j] .== z] .>= lb)) .&  (sim.A[Age = j][sim.Z[Age = j] .== z] .< ub) 
            vals = counter(sim.A[Age = j][sim.Z[Age = j] .== z][idx])
            # vals = unique(sim.A[Age = j][sim.Z[Age = j] .== z][idx])

            # We check that this set is not empty
            if !isempty(vals)
                λ[Age = j, Z = z, a = i] += 0.
                # w = Dict(key => (key - lb)./(ub - lb) for key ∈ keys(vals))
                w = [(key - lb)./(ub - lb) for key ∈ keys(vals)]
                # μ = Dict(key => vals[key]/N for key ∈ keys(vals))
                μ = [vals[key]/N for key ∈ keys(vals)]
                λ[Age = j, Z = z, a = i] += sum((1 .- w) .* μ)
                λ[Age = j, Z = z, a = i+1] += sum(w .* μ)
            end
        end
    end

    @assert sum(λ[Age = j]) ≈ 1.
    λ[Age = j] *= pm.μ[j]
end

λ2 = AxisArray(zeros(pm.a_size, pm.z_size, pm.J);
a = 1:pm.a_size,
Z = (pm.z_chain.state_values),
Age = 1:pm.J
)

# Si on veut calculer direct avec distrib de la pop prendre N qui compte tous les ages
N = size(filter(!isnan,sim.A), 1)
for j ∈ 1:pm.J
    for z ∈ pm.z_chain.state_values
        for (i, lb) ∈ enumerate(pm.a_vals)
            if lb == pm.a_vals[end]
                ub = Inf
            else
                ub = pm.a_vals[i + 1]
            end

            # We collect all assets in a certain interval of the grid
            idx = ((sim.A[Age = j][sim.Z[Age = j] .== z] .>= lb)) .&  (sim.A[Age = j][sim.Z[Age = j] .== z] .< ub) 
            vals = counter(sim.A[Age = j][sim.Z[Age = j] .== z][idx])
            # vals = unique(sim.A[Age = j][sim.Z[Age = j] .== z][idx])

            # We check that this set is not empty
            if !isempty(vals)
                λ2[Age = j, Z = z, a = i] += 0.
                # w = Dict(key => (key - lb)./(ub - lb) for key ∈ keys(vals))
                w = [(key - lb)./(ub - lb) for key ∈ keys(vals)]
                # μ = Dict(key => vals[key]/N for key ∈ keys(vals))
                μ = [vals[key]/N for key ∈ keys(vals)]
                λ2[Age = j, Z = z, a = i] += sum((1 .- w) .* μ)
                λ2[Age = j, Z = z, a = i+1] += sum(w .* μ)
            end
        end
    end
end
@assert sum(λ2) ≈ 1.

λ







plot(λ[Age = 60, Z = :E])
plot!(λ[Age = 54, Z = :E])
plot!(λ[Age = 50, Z = :E])
plot!(λ[Age = 44, Z = :E])
plot!(λ[Age = 1, Z = :E])

sum(λ) ≈ 1.
for key ∈ keys(vals)
    print(vals[key])
end

sim.A[Age = 1]

size(filter(!isnan,sim.A))

size(filter(!isnan,sim.A[Age = 56]), 1)/size(filter(!isnan,sim.A),1)

pl = plot()
xlabel!(L"Age")
ylabel!(L"Assets")
for n=1:3000
    plot!(pl, sim.A[N = n] , label=nothing, color=:red, alpha=0.1)
end
pl

λ.mean(axis=(2,3))
bar(dropdims(sum(dropdims(sum(λ, dims=3), dims=3), dims=2), dims=2))

A[Age = 1]

plot(dr.A[Age = 1])
plot(dr.C[Age = 45])
plot(dr.V[Age = 45])
