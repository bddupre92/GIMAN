using Test
using MechanisticTwin

@testset "Neuron Death Module" begin
    p = NeuronDeathParams()

    @testset "SBR observation model" begin
        # Full neuron count → baseline SBR
        sbr_full = sbr_observation(p.N_0, p)
        @test isapprox(sbr_full, p.SBR_0, rtol=0.01)

        # Half neurons → reduced SBR (sublinear due to gamma < 1)
        sbr_half = sbr_observation(p.N_0 / 2, p)
        @test sbr_half < sbr_full
        @test sbr_half > sbr_full / 2  # Sublinear: gamma=0.7 means less than proportional drop
    end

    @testset "Neuron count at diagnosis" begin
        N_diag = neuron_count_at_diagnosis(p)
        @test N_diag < p.N_0
        @test N_diag > 0
    end

    @testset "Toxicity function" begin
        # Zero aggregates → zero toxicity
        @test toxicity(0.0, 0.0, p) == 0.0

        # Oligomers more toxic than fibrils
        tox_O = toxicity(1.0, 0.0, p)
        tox_F = toxicity(0.0, 1.0, p)
        @test tox_O > tox_F  # alpha_tox > beta_tox

        # Linearity
        @test isapprox(toxicity(2.0, 0.0, p), 2.0 * tox_O)
    end

    @testset "ODE produces neuron decline" begin
        u = [p.N_0]
        du = similar(u)
        neuron_death_ode!(du, u, p, 0.0; O=1.0, F=0.5)
        @test du[1] < 0  # Neurons should be declining
    end

    @testset "solve_neuron_death returns N at requested times" begin
        # Aging-only baseline (no aggregates) over 5 years
        t_obs = [0.0, 1.0, 3.0, 5.0]
        N_t = solve_neuron_death(p, t_obs; O=0.0, F=0.0)
        @test length(N_t) == length(t_obs)
        @test N_t[1] ≈ p.N_0          # initial condition preserved
        @test all(diff(N_t) .< 0)     # monotonically decreasing
        # Aging at k_age=0.005/yr → ~2.5% loss over 5 years
        @test N_t[end] / p.N_0 ≈ exp(-p.k_age * 5.0) atol=1e-3
    end

    @testset "solve_neuron_death with toxicity is faster" begin
        t_obs = [0.0, 5.0]
        N_baseline = solve_neuron_death(p, t_obs; O=0.0, F=0.0)
        N_toxic    = solve_neuron_death(p, t_obs; O=10.0, F=5.0)
        @test N_toxic[end] < N_baseline[end]  # toxicity accelerates decline
    end

    @testset "sbr_loglikelihood is finite and well-formed" begin
        t_obs = [0.0, 2.0, 4.0]
        # Generate synthetic SBR observations under nonzero toxicity so that
        # k_death is identifiable (with O=F=0 the toxicity term vanishes and
        # k_death has no effect on the trajectory).
        O_true, F_true = 5.0, 2.0
        N_true = solve_neuron_death(p, t_obs; O=O_true, F=F_true)
        sbr_true = [sbr_observation(N, p) for N in N_true]
        ll = sbr_loglikelihood(p, t_obs, sbr_true; O=O_true, F=F_true)
        @test isfinite(ll)
        # log-likelihood at the data-generating params should beat a far-off
        # value of k_death.
        p_wrong = NeuronDeathParams(k_death=p.k_death * 10.0)
        ll_wrong = sbr_loglikelihood(p_wrong, t_obs, sbr_true; O=O_true, F=F_true)
        @test ll > ll_wrong
    end
end
