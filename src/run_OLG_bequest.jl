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
HHs = Households();
Policies = Policy()

λ = deserialize("data/distrib_03.dat")

function bequest_distrib(b_next, ϕ_next, age, ϕ, λ, Params)
    (; a_vals, j_star, ψ, a_min) = Params
    age_parents = age + 20
    if ϕ == :I
        # Already inherited so for sure no more bequest and state doesn't change 
        if (ϕ_next == :I) & (b_next == a_min)
            prob = 1.
        else
            prob = 0.
        end

    else  #if agent did not inherited yet 
        if (ϕ_next == :NI) & (b_next == a_min)
            # equal the probability that parents are still alive
            prob = ψ[age_parents]
        elseif (ϕ_next == :NI) & (b_next != a_min)
            # Can't have a positive bequest if did not inherit
            prob = 0.
        elseif ϕ_next == :I
            idx = findfirst(b_next .== a_vals)

            if age_parents < j_star
                χ = sum(λ.λ_a[Age = age_parents], dims = 2)[idx]
            else 
                χ = λ.λ_r[Age = age_parents - (j_star - 1)][idx]
            end
            prob = (1- ψ[age_parents]) * χ
        end
    end
    return prob
end
function compute_bequest_distrib!(υ::AxisArray, b_vals::Matrix, b_vals_index::Matrix, Params::NamedTuple)
    (; J) = Params
    for ϕ = unique(b_vals[:,2])
        for j = 1:J-20
            for (b_next_i, ϕ_next) ∈ zip(b_vals_index[:,1], b_vals[:,2])
                υ[b = b_next_i, ϕ_next = ϕ_next, ϕ = ϕ, Age = j] = bequest_distrib(b_vals[b_next_i,1], ϕ_next, j, ϕ, λ, Params) 
            end
        end
    end
    return υ
end

function bequest_distrib_(λ::NamedTuple, Params::NamedTuple)
    (; a_size, j_star, J, ψ) = Params
    υ = AxisArray(zeros(a_size, 2, 2, J-20);
        b = 1:a_size,
        ϕ_next = [:I, :NI],
        ϕ = [:I, :NI],
        Age = 1:J-20
    )

    for j ∈ 1:J-20
        age_parents = j + 20
        # Already inherited so for sure no more bequest and state doesn't change 
        υ[ϕ = :I, Age = j, ϕ_next = :I, b = 1] = 1.
        # Not inherited yet 
            ## Parents are still alive
            υ[ϕ = :NI, Age = j, ϕ_next = :NI, b = 1] = ψ[age_parents]
            ## Parents just dead
                if age_parents < j_star
                    ### Parents are not retired
                    υ[ϕ = :NI, Age = j, ϕ_next = :I] = (1 - ψ[age_parents]) .* sum(λ.λ_a[Age = age_parents], dims = 2)
                else
                    ### Parents are retired
                    υ[ϕ = :NI, Age = j, ϕ_next = :I] = (1 - ψ[age_parents]) .* λ.λ_r[Age = age_parents - (j_star - 1)]
                end
    end
    return υ
end


υ = AxisArray(zeros(HHs.a_size, 2, 2, HHs.J-20);
b = 1:HHs.a_size,
ϕ_next = [:I, :NI],
ϕ = [:I, :NI],
Age = 1:HHs.J-20
)

b_vals = gridmake(HHs.a_vals, [:I, :NI])
b_vals_index = gridmake(1:HHs.a_size, 1:2)



υ = compute_bequest_distrib!(υ, b_vals, b_vals_index, HHs);

sum(υ[ϕ = :I, Age = 35])
sum(λ.λ_r)

maximum(υ[ϕ = :I, ϕ_next= :NI])

ww = bequest_distrib_(λ, HHs)

ww .== υ