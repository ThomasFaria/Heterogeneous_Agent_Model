using Heterogenous_Agent_Model, QuantEcon, LaTeXStrings, Parameters, Plots, Serialization, StatsPlots, AxisArrays, Printf

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

Firm = Firms();
HHs = Households();

Results = Dict()
for θ ∈ range(0,1,11)
    Results[θ] = solve_equilibrium(
                                    4., 
                                    0.94 * HHs.h * sum(HHs.μ[1:HHs.j_star-1] .* HHs.ϵ),
                                    0.5,
                                    Firm,
                                    HHs,
                                    Policy(θ = θ), 
                                    η_tol_K=1e-3,
                                    η_tol_B=1e-3
                                )
    
end

serialize("data/Results.dat", Results)






######################################################
######################################################

## PLOTS

function plot_consumption_profiles(Results::Dict, θ::Float64, Params::NamedTuple)
    (; J) = Params
    C_a = reshape(sum(sum(Results[θ].λ.λ_a .* Results[θ].dr.Act.C, dims=2), dims=1), :, 1)
    C_r = reshape(sum(Results[θ].λ.λ_r .* Results[θ].dr.Ret.C, dims=1), :, 1)

    p = plot((1:J) .+ 20
        , vcat(C_a, C_r)
        , label=nothing
        )
    xlabel!(L"Age")
    ylabel!(L"Consumption")

    return p
end
plot_consumption_profiles(Results, 0.1, HHs)

function plot_wealth_profiles(Results::Dict, θ::Float64, Params::NamedTuple)
    (; J) = Params
    A_a = reshape(sum(sum(Results[θ].λ.λ_a .* Results[θ].dr.Act.A, dims=2), dims=1), :, 1)
    A_r = reshape(sum(Results[θ].λ.λ_r .* Results[θ].dr.Ret.A, dims=1), :, 1)

    p = plot((1:J) .+ 20
        , vcat(A_a, A_r)
        , label=nothing
        )
    xlabel!(L"Age")
    ylabel!(L"Asset")

    return p
end

function plot_wealth_profiles(Results::Dict, Policies::Vector{Float64}, Params::NamedTuple)
    (; J) = Params
    p = plot()

    for θ ∈ Policies
        A_a = reshape(sum(sum(Results[θ].λ.λ_a .* Results[θ].dr.Act.A, dims=2), dims=1), :, 1)
        A_r = reshape(sum(Results[θ].λ.λ_r .* Results[θ].dr.Ret.A, dims=1), :, 1)

        plot!(p
            , (1:J) .+ 20
            , vcat(A_a, A_r)
            , label= @sprintf("θ = %.1f", θ)
            )
    end

    xlabel!(L"Age")
    ylabel!(L"Asset")

    return p
end

plot_wealth_profiles(Results, 0.9, HHs)
plot_wealth_profiles(Results, [0.0,0.6, 0.9], HHs)

function plot_wealth_distrib(Results::Dict, θ::Float64, Params::NamedTuple)
    (; a_vals) = Params
    distrib = sum(sum(Results[θ].λ_scaled.λ_a, dims=2), dims=3) .+ sum(Results[θ].λ_scaled.λ_r, dims=2)


    p = bar(a_vals
        , reshape(distrib, :,1)
        , label=nothing
        )
    xlabel!(L"Asset")
    return p
end
plot_wealth_distrib(Results, 0.1, HHs)

function plot_wealth_by_age(Results::Dict, θ::Float64, ages::Vector{Int64},  Params::NamedTuple)
    (; a_vals, j_star) = Params

    p = plot()
    for j ∈ ages 
        true_age = j+20
        if j < j_star
            bar!(p
            , a_vals
            , sum(Results[θ].λ.λ_a[Age = j], dims=2)
            , label= @sprintf("%.0f ans", true_age)
            )
        else
            bar!(p
            , a_vals
            , Results[θ].λ.λ_r[Age = j - (j_star-1)]
            , label= @sprintf("%.0f ans", true_age)
            )

        end
    end

    xlabel!(L"Asset")
    return p
end
plot_wealth_by_age(Results, 0.3, [24, 30, 36, 40], HHs)
plot_wealth_by_age(Results, 0.3, [44, 50, 56, 60], HHs)
