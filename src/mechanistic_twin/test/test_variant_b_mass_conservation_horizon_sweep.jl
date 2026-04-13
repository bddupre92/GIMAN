using DifferentialEquations

k_prod    = 0.1
k_n       = 1.27e-3
k_e       = 0.09
k_conv    = 0.095
k_frag    = 0.01
k_clear_M = 0.05
k_clear_O = 0.02
k_clear_F = 0.005
k_age_hr  = 0.005 / (365.25 * 24.0)
alpha_tox = 1.53e-1

function cohen!(du, u, p, t)
    M, O, F, N = u
    Mc = max(M, 0.0); Oc = max(O, 0.0); Fc = max(F, 0.0); Nc = max(N, 1.0)
    du[1] = k_prod - k_n*Mc^2 - k_e*Mc*Fc - k_clear_M*Mc
    du[2] = k_n*Mc^2 - k_conv*Oc - k_clear_O*Oc
    du[3] = k_conv*Oc + k_frag*Fc - k_clear_F*Fc
    du[4] = -alpha_tox*Oc*Nc - k_age_hr*Nc
end

u0 = [k_prod/k_clear_M, 0.0, 1e-3, 400_000.0]

println("=== Devil's Advocate Test 1: Cohen-style at 1.5-year horizon ===")
println()
println("Sweep horizons:")
for years in [1.0, 1.5, 2.0, 3.0, 4.0]
    t_end_hr = years * 365.25 * 24.0
    prob = ODEProblem(cohen!, u0, (0.0, t_end_hr))
    t_start = time()
    sol = solve(prob, Rosenbrock23(); reltol=1e-8, abstol=1e-10, maxiters=Int(1e7))
    t_solve = time() - t_start
    F_end = sol.u[end][3]
    N_end = sol.u[end][4]
    n_steps = length(sol.t)
    println("  t = ", years, " yr:  solve = ", round(t_solve, digits=3), " s,  steps = ", n_steps, ",  F = ", round(F_end, sigdigits=3), " nM,  N = ", round(Int, N_end))
end

println()
println("Benchmarking 100 forward solves at 1.5 yr...")
years = 1.5
t_end_hr = years * 365.25 * 24.0
prob = ODEProblem(cohen!, u0, (0.0, t_end_hr))
# Warm up
solve(prob, Rosenbrock23(); reltol=1e-8, abstol=1e-10, maxiters=Int(1e7))
t = @elapsed begin
    for _ in 1:100
        solve(prob, Rosenbrock23(); reltol=1e-8, abstol=1e-10, maxiters=Int(1e7))
    end
end
per_solve_ms = t / 100 * 1000
println("  ", round(per_solve_ms, digits=2), " ms per solve")
println("  At ~5,600 evals per NUTS run (50 samples + 25 warmup): ~", round(per_solve_ms/1000 * 5600, digits=1), " s = ", round(per_solve_ms/1000 * 5600 / 60, digits=2), " min per patient")
println("  At 304 patients (Wave A): ~", round(per_solve_ms/1000 * 5600 * 304 / 3600, digits=1), " hours total")
