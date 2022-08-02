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

function get_r(K, L, Firm_pm::NamedTuple)
    (;α, A, δ) = Firm_pm
    return A * α * (L/K)^(1-α) - δ
end

function get_w(K, L, Firm_pm::NamedTuple)
    (;α, A) = Firm_pm
    return A * (1-α) * (K/L)^α
end


using Interpolations


function state_transition_OLG(z_next::Symbol, z::Symbol, Params::NamedTuple)
    (; z_chain) = Params

    # Get the index of the current/next idiosync shock in the transition matrix
    z_i = z_chain.state_values .== z
    next_z_i = z_chain.state_values .== z_next

    π = z_chain.p[z_i, next_z_i]

    return π[:][begin]
end

function get_state_value_index(mc::MarkovChain, value)
    idx = findall(mc.state_values .== value)
    return idx[begin]
end 

function c_transition_OLG(a::Float64, a_next::Vector{Float64}, z::Symbol, age::Int64,  Params::NamedTuple)
    (; r, q, z_chain) = Params
    idx = get_state_value_index(z_chain, z)

    c = (1 + r) * a + q[age, idx] - a_next[begin] 
    return c
end

function ubound(a::Float64, z::Symbol, age::Int64,  Params::NamedTuple)
    (; r, q, z_chain, a_min) = Params
    idx = get_state_value_index(z_chain, z)

    # Case when you save everything, no consumption only savings
    c =  a_min
    ub = (1 + r) * a + q[age, idx] - c
    ub = ifelse(ub <= a_min, a_min, ub)

    return [ub]
end

function obj_OLG(V::AxisArray{Float64, 3}, a_new::Vector{Float64}, a::Float64, z::Symbol, age::Int64, Params::NamedTuple)
    (; ψ,  u, β, z_chain, a_vals, β, u) = Params
    # Initialise the expectation
    Ev_new = 0
    for z_next ∈ z_chain.state_values
        π = state_transition_OLG(z_next, z, pm)
        if π == 0
            # No need to interpolate in this case, so we save some time
        else
            # # Extract the value function a skill given for the next shock
            v = V[Age = age + 1, Z = z_next]
            # Interpolate the value function on the asset for the following state
            v_new = CubicSplineInterpolation(a_vals, v, extrapolation_bc = Line())(a_new[begin])
            # Compute the expectation
            Ev_new += v_new * π
        end
    end
    
    c = c_transition_OLG(a, a_new, z, age,  Params)
    VF = u(c) + β* ψ[age + 1] * Ev_new
    return VF
end


V = AxisArray(zeros(pm.a_size, pm.z_size, pm.J+1);
                    a = 1:pm.a_size,
                    Z = ([:U, :E]),
                    Age = 1:pm.J+1
            )

A = AxisArray(zeros(pm.a_size, pm.z_size, pm.J+1);
                    a = 1:pm.a_size,
                    Z = ([:U, :E]),
                    Age = 1:pm.J+1
            )

Tv = AxisArray(zeros(pm.a_size, pm.z_size, pm.J+1);
            a = 1:pm.a_size,
            Z = ([:U, :E]),
            Age = 1:pm.J+1
    )

C = AxisArray(zeros(pm.a_size, pm.z_size, pm.J+1);
    a = 1:pm.a_size,
    Z = ([:U, :E]),
    Age = 1:pm.J+1
)


using ProgressBars, Optim

# Loop over ages recursively
iter = ProgressBar(pm.J:-1:1)
for j ∈ iter
    # Loop over assets
    for a_i ∈ 1:pm.a_size
        a = pm.a_vals[a_i]

        # Loop over idiosync shock
        for z ∈ pm.z_chain.state_values
            # Specify lower bound for optimisation
            lb = [pm.a_min]        
            # Specify upper bound for optimisation
            ub = ubound(a, z, j, pm)
            # Specify the initial value in the middle of the range
            init = (lb + ub)/2

            # Optimization
            # if lb != ub
                Sol = optimize(x -> -obj_OLG(V, x, a, z, j, pm), lb, ub, init)
                a_new = Optim.minimizer(Sol)
            # end
            # Deduce optimal value function and consumption
            c = c_transition_OLG(a, a_new, z, j,  pm)

            Tv[Age = j, Z = z, a = a_i] = obj_OLG(V, a_new, a, z, j, pm)
            C[Age = j, Z = z, a = a_i] = c
            A[Age = j+1, Z = z, a = a_i] = a_new[begin]

        end
    end
end



A[Age = 45]
C[Age = 65]
Tv[Age = 65]


age = 4               
z = :E     
a_new = [1.]
a = 0.98





# Specify lower bound for optimisation
lb = [pm.a_min]   
# Specify upper bound for optimisation
ub = ubound(a, z, age, pm)
# Specify the initial value in the middle of the range
init = (lb + ub)/2

# Optimization
if lb != ub
    Sol = optimize(x -> -obj_OLG(V, x, a, z, age, pm), lb, ub, init)
    a_new = Optim.minimizer(Sol)
end

obj_OLG(V, [0.0000000009], a, z, age, pm)

using Optim








