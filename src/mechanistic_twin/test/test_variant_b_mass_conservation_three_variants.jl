using DifferentialEquations

# Devil's Advocate Test 2: Hill-saturating fibril clearance
#
# Tanik 2013 finding: fibrils sabotage their own clearance machinery.
# Mathematical encoding (Mathematical Biologist's recommendation):
#   k_clear_F_eff(F) = k_clear_F_max / (1 + (F/K_F)^n)  -- WRONG: this DECREASES clearance
# Wait — re-reading Tanik 2013: fibrils impair clearance. So as F increases, clearance
# DECREASES. That's an ANTI-saturation, which makes things WORSE.
#
# Reframing: what we actually want is that clearance has a HARD CAP at high F (Tanik
# observation: clearance machinery is overwhelmed but the rate doesn't go to zero
# instantly). This is Michaelis-Menten:
#   removal_rate = V_max * F / (K_M + F)
#
# At low F, this looks like first-order with rate V_max/K_M. At high F, it saturates
# at V_max regardless of F. So the maximum amount of F that can be cleared per unit
# time is V_max — and for the system to be stable we need V_max > k_conv*O*max + k_frag*F
# at the F we care about.
#
# This still doesn't fix unbounded growth in F itself; it just caps the linear sink.
# The REAL fix is: there must be a CAPACITY term on the F production side.
# Physically: there's only so much α-syn substrate available to be converted to fibrils.
# The intracellular α-syn pool is bounded by k_prod / k_clear_M ~ 2 nM. So F production
# via k_conv*O is bounded above by k_conv * (k_prod/k_clear_M)^2 / (k_conv + k_clear_O) =
# 0.095 * 4 / 0.115 = 3.3 nM/hr at MOST. This is a FINITE rate, not unbounded.
#
# So why does F blow up? Because k_frag*F is a POSITIVE feedback in F that has no
# upper bound. The Cohen 2013 framework's k_frag is a fragmentation rate that creates
# new fibril ENDS (number concentration P), not new fibril MASS (M). When you write
# the mass equation as `dF/dt = ... + k_frag*F`, you've conflated end-creation with
# mass-creation. That's a units error.
#
# CORRECT FIX: REMOVE k_frag from the mass equation. Fragmentation creates new ends
# but conserves mass. Then F grows only via k_conv*O, which is bounded above by the
# monomer reservoir.
#
# Let me test BOTH variants:
#   (a) Remove k_frag from dF/dt mass equation (the fundamental fix)
#   (b) Hill-cap k_clear_F (the band-aid)
#   (c) Both

println("=== Devil's Advocate Test 2: Three Cohen-style variants ===")
println()

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

# Variant A: Original (broken) — k_frag in mass equation
function cohen_A!(du, u, p, t)
    M, O, F, N = u
    Mc = max(M, 0.0); Oc = max(O, 0.0); Fc = max(F, 0.0); Nc = max(N, 1.0)
    du[1] = k_prod - k_n*Mc^2 - k_e*Mc*Fc - k_clear_M*Mc
    du[2] = k_n*Mc^2 - k_conv*Oc - k_clear_O*Oc
    du[3] = k_conv*Oc + k_frag*Fc - k_clear_F*Fc
    du[4] = -alpha_tox*Oc*Nc - k_age_hr*Nc
end

# Variant B: Remove k_frag from mass equation (the fundamental fix)
function cohen_B!(du, u, p, t)
    M, O, F, N = u
    Mc = max(M, 0.0); Oc = max(O, 0.0); Fc = max(F, 0.0); Nc = max(N, 1.0)
    du[1] = k_prod - k_n*Mc^2 - k_e*Mc*Fc - k_clear_M*Mc
    du[2] = k_n*Mc^2 - k_conv*Oc - k_clear_O*Oc
    du[3] = k_conv*Oc - k_clear_F*Fc        # k_frag REMOVED — fragmentation conserves mass
    du[4] = -alpha_tox*Oc*Nc - k_age_hr*Nc
end

# Variant C: B + Tanik 2013 capacity feedback (clearance impairment as F grows)
# k_clear_F_eff = k_clear_F * F_threshold / (F_threshold + F)
# This DECREASES clearance as F grows. Tanik observation. But it MUST be combined
# with the mass-conserving fragmentation (variant B) or else F still blows up.
F_threshold = 100.0  # nM, intracellular Lewy body packing limit
function cohen_C!(du, u, p, t)
    M, O, F, N = u
    Mc = max(M, 0.0); Oc = max(O, 0.0); Fc = max(F, 0.0); Nc = max(N, 1.0)
    du[1] = k_prod - k_n*Mc^2 - k_e*Mc*Fc - k_clear_M*Mc
    du[2] = k_n*Mc^2 - k_conv*Oc - k_clear_O*Oc
    k_clear_F_eff = k_clear_F * F_threshold / (F_threshold + Fc)  # Tanik: clearance impaired
    du[3] = k_conv*Oc - k_clear_F_eff*Fc
    du[4] = -alpha_tox*Oc*Nc - k_age_hr*Nc
end

u0 = [k_prod/k_clear_M, 0.0, 1e-3, 400_000.0]

for (name, ode_fn) in [("A: Original (with k_frag in mass eq)", cohen_A!),
                       ("B: k_frag removed from mass eq", cohen_B!),
                       ("C: B + Tanik clearance impairment", cohen_C!)]
    println("--- Variant ", name, " ---")
    for years in [1.0, 2.0, 4.0, 10.0]
        t_end_hr = years * 365.25 * 24.0
        prob = ODEProblem(ode_fn, u0, (0.0, t_end_hr))
        t_start = time()
        sol = solve(prob, Rosenbrock23(); reltol=1e-8, abstol=1e-10, maxiters=Int(1e7))
        t_solve = time() - t_start
        F_end = sol.u[end][3]
        N_end = sol.u[end][4]
        println("  t = ", years, " yr:  ", round(t_solve*1000, digits=1), " ms,  F = ", round(F_end, sigdigits=4), " nM,  N = ", round(Int, N_end))
    end
    println()
end

println("=== Per-solve benchmark for Variant B (4 yr) ===")
prob = ODEProblem(cohen_B!, u0, (0.0, 4*365.25*24.0))
solve(prob, Rosenbrock23(); reltol=1e-8, abstol=1e-10, maxiters=Int(1e7))  # warm
t = @elapsed begin
    for _ in 1:100
        solve(prob, Rosenbrock23(); reltol=1e-8, abstol=1e-10, maxiters=Int(1e7))
    end
end
per_solve_ms = t / 100 * 1000
println("  ", round(per_solve_ms, digits=2), " ms per solve at 4 yr horizon")
println("  At ~5,600 evals/NUTS run: ~", round(per_solve_ms/1000 * 5600, digits=1), " s = ", round(per_solve_ms/1000 * 5600 / 60, digits=2), " min/patient")
println("  At 304 patients: ~", round(per_solve_ms/1000 * 5600 * 304 / 3600, digits=1), " hours total")
