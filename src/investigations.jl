C = dot(x.res.λ_scaled.λ_a, x.res.dr.Act.C) + dot(x.res.λ_scaled.λ_r, x.res.dr.Ret.C)
K = dot(x.res.λ_scaled.λ_a, x.res.dr.Act.A) + dot(x.res.λ_scaled.λ_r, x.res.dr.Ret.A)
C + x.K * Firm.δ
Y = (Firm.δ + x.r) * K + x.w * L

Firm.Ω * x.K^Firm.α * L^(1 - Firm.α)

Firm = Firms(Ω=1.3193);

kk = 5.2241
yy = 1.22
cc = 0.7396
rr = 0.0041
ww = 1.0064

1.3193 * kk^Firm.α * 0.35^(1 - Firm.α)

(Firm.δ + rr) * kk + ww/(mean(HHs.ϵ)*HHs.h) * 0.35

kk = 4.0599
yy = 1.1140
cc = 0.7409
rr = 0.0192
ww = 0.9170
tau = 0.0609
popo = Policy(θ = 0.3)

r_to_w(rr, Firm) * (1 - tau - popo.τ_u) * HHs.h * mean(HHs.ϵ)

1.3193 * kk^Firm.α * 0.35^(1 - Firm.α)
(Firm.δ + rr) * kk + ww/(mean(HHs.ϵ)*HHs.h* (1 - tau - popo.τ_u)) * 0.35

Firm.Ω * x1.K^Firm.α * L^(1 - Firm.α)
(Firm.δ + x1.r) * x1.K + x1.w * L
x1.w * (mean(HHs.ϵ) * HHs.h * (1 - Policies.τ_ssc - Policies.τ_u))

sum(x1.dr.Act.C .* x1.λ.λ_a) + sum(x1.dr.Ret.C .* x1.λ.λ_r) + Firm.δ * x1.K + x1.B

sum(dropdims(sum(x1.λ.λ_a, dims=1), dims=1)' .* (q[Age=1:44] .+ x1.B)) + sum(x1.λ.λ_r * (q[Age=45, Z=:U] + x1.B))

Firm.δ * kk + cc

x1.dr.Ret.A[Age=1]
q = get_dispo_income(x1.w, HHs, Policies)
using LinearAlgebra

dot(dropdims(sum(x1.λ.λ_a, dims=1), dims=1)', q[1:44,:]) + sum(x1.λ.λ_r) * q[end,end]

Firm.Ω * x1.K^Firm.α * L^(1 - Firm.α)

x1.w


(1+x1.r) * HHs.a_vals[52] - x1.dr.Ret.A[Age=3, a=52] + q[Age = 47, Z=:E] + x1.B

x1.dr.Ret.C[Age=3, a=52]
C_r[Age=3, a=52]


C_a = AxisArray(zeros(HHs.a_size, HHs.z_size,  HHs.j_star-1);
a = 1:HHs.a_size,
Z = (HHs.z_chain.state_values),
Age = 1:HHs.j_star-1
)

C_r = AxisArray(zeros(HHs.a_size, HHs.J-(HHs.j_star-1));
a = 1:HHs.a_size,
Age = HHs.j_star:HHs.J
)

for j ∈ 1:HHs.j_star-1
    for z ∈ HHs.z_chain.state_values
        C_a[Age = j, Z=z] = (1+x1.r) * HHs.a_vals .- x1.dr.Act.A[Age = j, Z=z] .+ q[Age = j, Z=z] .+ x1.B
    end
end

for j ∈ HHs.j_star:HHs.J
   C_r[Age = j - (HHs.j_star-1)] = (1+x1.r) * HHs.a_vals .- x1.dr.Ret.A[Age = j - (HHs.j_star-1)] .+ q[Age = j, Z=:U] .+ x1.B
end


x1.K
C = dot(x1.λ.λ_a, x1.dr.Act.C) + dot(x1.λ.λ_r, x1.dr.Ret.C)
C2 = dot(x1.λ.λ_a, C_a) + dot(x1.λ.λ_r, C_r)
K = dot(x1.λ.λ_a, x1.dr.Act.A) + dot(x1.λ.λ_r, x1.dr.Ret.A)
C2 + x1.K * Firm.δ

Y = (Firm.δ + x1.r) * K + x1.w * L



(1+x1.r) * HHs.a_vals[52] - x1.dr.Act.A[Age=30, a=52, Z=:E] + q[Age = 30, Z=:E] + x1.B

x1.dr.Act.C[Age=30, a=52, Z=:E]
C_a[Age=30, a=52, Z=:E]

q = get_dispo_income(x1.w, HHs, Policies)

(1+r) * HHs.a_vals[52] - dr.Act.A[Age=30, a=52, Z=:E] + q[Age = 30, Z=:E] + B

dr.Act.C[Age=30, a=52, Z=:E]



C_a = AxisArray(zeros(HHs.a_size, HHs.z_size,  HHs.j_star-1);
    a = 1:HHs.a_size,
    Z = (HHs.z_chain.state_values),
    Age = 1:HHs.j_star-1
)

z=:E
age_i = 1

C_a[Age = age_i, Z=z]

for 
(1+x1.r) * HHs.a_vals .- x1.dr.Act.A[Age = age_i, Z=z] .+ q[Age = age_i, Z=z] .+ x1.B

(1+x1.r) * HHs.a_vals - A_r[Age = age_i, a = a_past_i] + q[Age = j, Z=:U] + B