using Heteregenous_Agent_Model, QuantEcon, LaTeXStrings, Parameters, Plots

Params = @with_kw (
                    r = 0.04, # interest rate
                    σ = 1.5, # Constant relative risk aversion (consumption utility)
                    Σ = 1., # Constant relative risk aversion (asset utility)
                    β = 0.96,
                    ϵ = 1e-5,
                    z_chain = MarkovChain(
                                            [0.9 0.1;
                                            0.1 0.9], 
                                        [1.0; 1.1]),
                    skill_chain = MarkovChain(
                                            [1. 0.;
                                             0. 1.], 
                                        [0.5; 1.0]),
                    age_chain = MarkovChain(
                                            [0. 1. 0.; 
                                            0.15 0. 0.85; 
                                            1. 0. 0.], 
                                        [1; 2; 3]),
                    a_min = 1e-10,
                    a_max = 5.0,
                    a_size = 100,
                    a_vals = range(a_min, a_max, length = a_size),
                    z_size = length(z_chain.state_values),
                    skill_size = length(skill_chain.state_values),
                    age_size = length(age_chain.state_values),
                    n = a_size * z_size * skill_size * age_size,
                    s_vals = gridmake(a_vals, z_chain.state_values, skill_chain.state_values, age_chain.state_values),
                    s_vals_index = gridmake(1:a_size, 1:z_size, 1:skill_size, 1:age_size),
                    u = σ == 1 ? c -> log(c) : c -> (c^(1 - σ)) / (1 - σ),
                    U = Σ == 1 ? a -> log(a) : a -> (a^(1 - Σ)) / (1 - Σ),
)

Model = Params(a_size = 5)
C = ones(Model.n) * Model.a_min
V = zeros(Model.n); 

eval_value_function(V, C, Model)

x = bellman_update(V, Model)

sol = solve_PFI(Model)

z_vals = Model.z_chain.state_values
skill_vals = Model.skill_chain.state_values
age_vals = Model.age_chain.state_values

plot( [i for i in Model.a_vals], sol.C[(Model.s_vals[:,2] .== z_vals[1]) .&& (Model.s_vals[:,3] .== skill_vals[1]) .&& (Model.s_vals[:,4] .== age_vals[1])],
    label = L"z = %$(z_vals[1])",legend=:bottomright)
plot!([i for i in Model.a_vals], sol.C[(Model.s_vals[:,2] .== z_vals[2]) .&& (Model.s_vals[:,3] .== skill_vals[1]) .&& (Model.s_vals[:,4] .== age_vals[1])], 
    label = L"z = %$(z_vals[2])",legend=:bottomright)
xlabel!(L"Assets")
ylabel!(L"Consumption")
title!(L"Decision \: rule")

plot( [i for i in Model.a_vals], sol.V[(Model.s_vals[:,2] .== z_vals[1]) .&& (Model.s_vals[:,3] .== skill_vals[1]) .&& (Model.s_vals[:,4] .== age_vals[1])],
    label = L"z = %$(z_vals[1])",legend=:bottomright)
plot!([i for i in Model.a_vals], sol.V[(Model.s_vals[:,2] .== z_vals[2]) .&& (Model.s_vals[:,3] .== skill_vals[1]) .&& (Model.s_vals[:,4] .== age_vals[1])],
    label = L"z = %$(z_vals[2])",legend=:bottomright)
xlabel!(L"Assets")
ylabel!(L"Value function")
title!(L"Decision \: rule")