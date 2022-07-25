using Heterogenous_Agent_Model, QuantEcon, LaTeXStrings, Parameters, Plots, Serialization, StatsPlots, AxisArrays


Params = @with_kw (
                    r = 0.04, # interest rate
                    σ = 1.1, # Constant relative risk aversion (consumption utility)
                    Σ = 1., # Constant relative risk aversion (asset utility)
                    β = 0.92,
                    ϵ = 1e-5,
                    z_chain = MarkovChain(
                                            [0.9 0.1;
                                            0.1 0.9], 
                                        [0.1; 1.]),
                    skill_chain = MarkovChain(
                                            [1. 0.;
                                            0. 1.], 
                                        [0.5; 1.0]),
                    age_chain = MarkovChain(
                                            [0. 1. 0. 0. 0.; 
                                            0.05 0. 0.95 0. 0.; 
                                            0.10 0. 0. 0.9  0.; 
                                            0.20 0. 0. 0. 0.8; 
                                            1. 0. 0. 0. 0.], 
                                            collect(1:5)),
                    a_min = 1e-10,
                    a_max = 10.0,
                    a_size = 70,
                    a_vals = range(a_min, a_max, length = a_size),
                    z_size = length(z_chain.state_values),
                    skill_size = length(skill_chain.state_values),
                    age_size = length(age_chain.state_values),
                    n = a_size * z_size * skill_size * age_size,
                    s_vals = gridmake(a_vals, z_chain.state_values, skill_chain.state_values, age_chain.state_values),
                    s_vals_index = gridmake(1:a_size, 1:z_size, 1:skill_size, 1:age_size),
                    u = σ == 1 ? c -> log(c) : c -> (c^(1 - σ)) / (1 - σ),
                    U = Σ == 1 ? a -> log(a) : a -> (a^(1 - Σ)) / (1 - Σ),
                    Ω = Dict(zip(age_chain.state_values, [0.5; 0.75; 1.; 1.1; 0.])),
);

Model = Params();
C = ones(Model.n) * Model.a_min;
V = zeros(Model.n); 

eval_value_function(V, C, Model);
bell = bellman_update(V, Model);
sol = solve_PFI(Model)

DR = (  V = AxisArray(reshape(sol.V, (Model.a_size, Model.z_size, Model.skill_size, Model.age_size));
                    a = 1:Model.a_size,
                    Z = ([:Low, :High]),
                    Skill = ([:Low, :High]),
                    Age = 1:Model.age_size),

        C = AxisArray(reshape(sol.C, (Model.a_size, Model.z_size, Model.skill_size, Model.age_size));
                    a = 1:Model.a_size,
                    Z = ([:Low, :High]),
                    Skill = ([:Low, :High]),
                    Age = 1:Model.age_size)
)

DR.C[Z = :Low, Skill = :Low, Age = 1]


z_vals = Model.z_chain.state_values
skill_vals = Model.skill_chain.state_values
age_vals = Model.age_chain.state_values

a_star = AxisArray(reshape(get_asset_from_dr(sol.C, Model), (Model.a_size, Model.z_size, Model.skill_size, Model.age_size));
                    a = 1:Model.a_size,
                    Z = ([:Low, :High]),
                    Skill = ([:Low, :High]),
                    Age = 1:Model.age_size
                    );

### Plot the next period asset in function of the current asset
plot(Model.a_vals, a_star[Z = :Low, Skill = :Low, Age = 1], 
labels = L"z = Low", lw = 2, alpha = 0.6, legend=:bottomright)
plot!(Model.a_vals, a_star[Z = :High, Skill = :Low, Age = 1], 
labels = L"z = High", lw = 2, alpha = 0.6)
plot!(Model.a_vals, Model.a_vals, label = "", color = :black, linestyle = :dash)
plot!(xlabel = "current assets", ylabel = "next period assets")

plot( Model.a_vals, DR.C[Z = :Low, Skill = :Low, Age = 1],
    label = L"z = Low",legend=:bottomright)
plot!(Model.a_vals, DR.C[Z = :High, Skill = :Low, Age = 1], 
    label = L"z = High",legend=:bottomright)
xlabel!(L"Assets")
ylabel!(L"Consumption")
title!(L"Decision \: rule")

plot( Model.a_vals, DR.V[Z = :Low, Skill = :Low, Age = 1],
    label = L"z = Low",legend=:bottomright)
plot!(Model.a_vals, DR.V[Z = :High, Skill = :Low, Age = 1], 
    label = L"z = High",legend=:bottomright)
xlabel!(L"Assets")
ylabel!(L"Value function")
title!(L"Decision \: rule")

N = 1000
sim = simulate_model(sol.C, Model, N=N, a0=0.);
pl = plot()
xlabel!(L"Periods")
ylabel!(L"Assets")
for n=1:N
    # We plot only the high skilled individuals
    if sim[n,1,3] == 1.0
        plot!(pl, sim[n,1:40,1], label=nothing, color=:red, alpha=0.1)
    end
end
pl

# Low skill
StatsPlots.density(sim[:,1000,1][sim[:,1000,3] .== 0.5], xlims=(0,4), label=nothing)
StatsPlots.density!(sim[3,40:1000,1], label=nothing)
xlabel!(L"Assets")
title!(L"Distribution \: of \: assets  \: for  \: low  \: skilled")


# High 
StatsPlots.density(sim[:,1000,1][sim[:,1000,3] .== 1.0], xlims=(0,7), label=nothing)
StatsPlots.density!(sim[1,40:1000,1], label=nothing)
xlabel!(L"Assets")
title!(L"Distribution \: of \: assets  \: for  \: high  \: skilled")


# Both 
StatsPlots.density(sim[:,1000,1][sim[:,1000,3] .== 0.5], label = L"w = 0.5", xlims=(0,7))
StatsPlots.density!(sim[:,1000,1][sim[:,1000,3] .== 1.0], label = L"w = 1.0")
xlabel!(L"Assets")
title!(L"Distribution \: of \: assets  \: in  \: function  \: of \: skill")

# Density over the idiosync shock
StatsPlots.density(sim[1, 40:1000, 1], labels = L"All", xlims=(0,7))
StatsPlots.density!(sim[1,40:1000, 1][sim[1, 40:1000, 2] .== 0.1], labels = L"z=0.1")
StatsPlots.density!(sim[1,40:1000, 1][sim[1, 40:1000, 2] .== 1.0], labels = L"z=1.0")
xlabel!(L"Assets")
title!(L"Distribution \: of \: assets  \: in  \: function  \: of \: idiosyncratic \: shock")

# Density over the ages
StatsPlots.density(sim[1, 40:1000, 1], labels = L"All", xlims=(0,7))
StatsPlots.density!(sim[1,40:1000, 1][sim[2, 40:1000, 4] .== 1.], labels = L"Age=1")
StatsPlots.density!(sim[1,40:1000, 1][sim[1, 40:1000, 4] .== 2.], labels = L"Age=2")
StatsPlots.density!(sim[1,40:1000, 1][sim[1, 40:1000, 4] .== 3.], labels = L"Age=3")
xlabel!(L"Assets")
title!(L"Distribution \: of \: assets  \: in  \: function  \: of \: age")



# TODO: Comprendre pourquoi il n'y a pas d'effet avec l'age? Comment modéliser les différences d'age ? 
# Faire ca !! 
# TODO: Pourquoi mon code est si long, qu'est ce que je fais de si peu optimal ?